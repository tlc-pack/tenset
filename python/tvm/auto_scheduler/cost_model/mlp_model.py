from collections import OrderedDict
import copy
from itertools import chain
import multiprocessing
import os
import pickle
import random
import time
import io
import json
import numpy as np
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger("auto_scheduler")

from tvm.auto_scheduler.dataset import Dataset, LearningTask
from tvm.auto_scheduler.feature import (
    get_per_store_features_from_measure_pairs, get_per_store_features_from_states)
from tvm.auto_scheduler.measure_record import RecordReader
from .xgb_model import get_workload_embedding
from .cost_model import PythonBasedModel


class SegmentDataLoader:
    def __init__(
            self,
            dataset,
            batch_size,
            device,
            use_workload_embedding=True,
            use_target_embedding=False,
            target_id_dict={},
            fea_norm_vec=None,
            shuffle=False,
    ):
        self.device = device
        self.shuffle = shuffle
        self.number = len(dataset)
        self.batch_size = batch_size

        self.segment_sizes = torch.empty((self.number,), dtype=torch.int32)
        self.labels = torch.empty((self.number,), dtype=torch.float32)

        # Flatten features
        flatten_features = []
        ct = 0
        for task in dataset.features:
            throughputs = dataset.throughputs[task]
            self.labels[ct: ct + len(throughputs)] = torch.tensor(throughputs)
            task_embedding = None
            if use_workload_embedding or use_target_embedding:
                task_embedding = np.zeros(
                    (10 if use_workload_embedding else 0),
                    dtype=np.float32,
                )

                if use_workload_embedding:
                    tmp_task_embedding = get_workload_embedding(task.workload_key)
                    task_embedding[:9] = tmp_task_embedding

                if use_target_embedding:
                    target_id = target_id_dict.get(
                        str(task.target), np.random.randint(0, len(target_id_dict))
                    )
                    task_embedding[9+target_id] = 1.0


            for row in dataset.features[task]:
                self.segment_sizes[ct] = len(row)

                if task_embedding is not None:
                    tmp = np.tile(task_embedding, (len(row), 1))
                    flatten_features.extend(np.concatenate([row, tmp], axis=1))
                else:
                    flatten_features.extend(row)
                ct += 1

        max_seg_len = self.segment_sizes.max()
        self.features = torch.tensor(np.array(flatten_features, dtype=np.float32))
        if fea_norm_vec is not None:
            self.normalize(fea_norm_vec)

        self.feature_offsets = (
                    torch.cumsum(self.segment_sizes, 0, dtype=torch.int32) - self.segment_sizes).cpu().numpy()
        self.iter_order = self.pointer = None

    def normalize(self, norm_vector=None):
        if norm_vector is None:
            norm_vector = torch.ones((self.features.shape[1],))
            for i in range(self.features.shape[1]):
                max_val = self.features[:, i].max().item()
                if max_val > 0:
                    norm_vector[i] = max_val
        self.features /= norm_vector

        return norm_vector

    def __iter__(self):
        if self.shuffle:
            self.iter_order = torch.randperm(self.number)
        else:
            self.iter_order = torch.arange(self.number)
        self.pointer = 0

        return self

    def sample_batch(self, batch_size):
        raise NotImplemented
        batch_indices = np.random.choice(self.number, batch_size)
        return self._fetch_indices(batch_indices)

    def __next__(self):
        if self.pointer >= self.number:
            raise StopIteration

        batch_indices = self.iter_order[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self._fetch_indices(batch_indices)

    def _fetch_indices(self, indices):
        segment_sizes = self.segment_sizes[indices]

        feature_offsets = self.feature_offsets[indices]
        feature_indices = np.empty((segment_sizes.sum(),), dtype=np.int32)
        ct = 0
        for offset, seg_size in zip(feature_offsets, segment_sizes.numpy()):
            feature_indices[ct: ct + seg_size] = np.arange(offset, offset + seg_size, 1)
            ct += seg_size

        features = self.features[feature_indices]
        labels = self.labels[indices]
        return (x.to(self.device) for x in (segment_sizes, features, labels))

    def __len__(self):
        return self.number


class SegmentSumMLPModule(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_norm=False, add_sigmoid=False):
        super().__init__()

        self.segment_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.add_sigmoid = add_sigmoid

        if use_norm:
            self.norm = torch.nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = torch.nn.Identity()

        self.l0 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Linear(hidden_dim, out_dim)

    def freeze_for_fine_tuning(self):
        for x in self.segment_encoder.parameters():
            x.requires_grad_(False)

    def forward(self, segment_sizes, features, params=None):
        n_seg = segment_sizes.shape[0]
        device = features.device

        segment_sizes = segment_sizes.long()

        features = self.segment_encoder(
            features
        )
        segment_indices = torch.repeat_interleave(
            torch.arange(n_seg, device=device), segment_sizes
        )

        n_dim = features.shape[1]
        segment_sum = torch.scatter_add(
            torch.zeros((n_seg, n_dim), dtype=features.dtype, device=device),
            0,
            segment_indices.view(-1, 1).expand(-1, n_dim),
            features,
        )
        output = self.norm(segment_sum)
        output = self.l0(output) + output
        output = self.l1(output) + output
        output = self.decoder(
            output
        ).squeeze()

        if self.add_sigmoid:
            output = torch.sigmoid(output)

        return output

class LSTMModuel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.segment_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.norm = torch.nn.Identity()

        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim)
        self.l0 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Linear(hidden_dim, out_dim)


    def forward(self, segment_sizes, features, params=None):
        features = self.segment_encoder(
            features
        )

        seqs = []
        ct = 0
        for seg_size in segment_sizes:
            seqs.append(features[ct: ct + seg_size])
            ct += seg_size
        output = torch.nn.utils.rnn.pad_sequence(seqs)

        output, (h, c)  = self.lstm(output)
        output = self.norm(h[0])
        output = self.l0(output) + output
        output = self.l1(output) + output

        output = self.decoder(
            output
        ).squeeze()

        return output



class MHAModule(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, out_dim, add_sigmoid=False):
        super().__init__()

        self.add_sigmoid = add_sigmoid

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.l0 = torch.nn.MultiheadAttention(hidden_dim, num_heads)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, segment_sizes, features):
        n_seg = segment_sizes.shape[0]
        device = features.device

        features = self.encoder(features)

        seqs = []
        ct = 0
        for seg_size in segment_sizes:
            seqs.append(features[ct: ct + seg_size])
            ct += seg_size
        output = torch.nn.utils.rnn.pad_sequence(seqs)

        output = self.l0(output, output, output)[0] + output
        output = self.decoder(output).sum(0).squeeze()

        if self.add_sigmoid:
            output = torch.sigmoid(output)

        return output


def make_net(params):
    if params["type"] == "SegmentSumMLP":
        return SegmentSumMLPModule(
            params["in_dim"], params["hidden_dim"], params["out_dim"],
            add_sigmoid=params['add_sigmoid']
        )
    elif params["type"] == "MultiHeadAttention":
        return MHAModule(
            params['in_dim'], params['hidden_dim'], params['num_heads'], params['out_dim'],
            add_sigmoid=params['add_sigmoid']
        )
    elif params["type"] == "LSTM":
        return LSTMModuel(
            params["in_dim"], params["hidden_dim"], params["out_dim"],
        )
    else:
        raise ValueError("Invalid type: " + params["type"])


def moving_average(average, update):
    if average is None:
        return update
    else:
        return average * 0.95 + update * 0.05


class MLPModelInternal:
    def __init__(self, device=None, few_shot_learning="base_only", use_workload_embedding=True, use_target_embedding=False,
                 loss_type='lambdaRankLoss'):
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda:0'
            else:
                device = 'cpu'
        print(device)
        # Common parameters
        self.net_params = {
            "type": "SegmentSumMLP",
            "in_dim": 164 + (10 if use_workload_embedding else 0),  
            "hidden_dim": 256,
            "out_dim": 1,
        }

        # self.net_params = {
        #    "type": "MultiHeadAttention",
        #    "in_dim": 164,
        #    "num_heads": 8,
        #    "hidden_dim": 1024,
        #    "out_dim": 1,
        # }

        self.target_id_dict = {}
        self.loss_type = loss_type
        self.n_epoch = 100
        self.lr = 7e-4
        

        if loss_type == 'rmse':
            self.loss_func = torch.nn.MSELoss()
            self.net_params['add_sigmoid'] = True
        elif loss_type == 'rankNetLoss':
            self.loss_func = RankNetLoss()
            self.net_params['add_sigmoid'] = False
            self.n_epoch = 30
        elif loss_type == 'lambdaRankLoss':
            self.loss_func = LambdaRankLoss()
            self.net_params['add_sigmoid'] = False
            self.lr = 7e-4
            self.n_epoch = 50
        elif loss_type == 'listNetLoss':
            self.loss_func = ListNetLoss()
            self.lr = 9e-4
            self.n_epoch = 50
            self.net_params['add_sigmoid'] = False
        else:
            raise ValueError("Invalid loss type: " + loss_type)

        self.grad_clip = 0.5
        self.few_shot_learning = few_shot_learning
        self.fea_norm_vec = None
        self.use_workload_embedding = use_workload_embedding
        self.use_target_embedding = use_target_embedding

        # Hyperparameters for self.fit_base
        self.batch_size = 512
        self.infer_batch_size = 4096
        self.wd = 1e-6
        self.device = device
        self.print_per_epoches = 5

        # Hyperparameters for MAML
        self.meta_outer_lr = 7e-4
        self.meta_inner_lr = 1e-2
        self.meta_test_num_steps = 5
        self.few_shot_number = 32
        self.meta_batch_size_tasks = 8
        self.meta_batch_size_per_task = 256

        # Hyperparameters for fine-tuning
        self.fine_tune_lr = 4e-2
        self.fine_tune_batch_size = 512
        self.fine_tune_num_steps = 10
        self.fine_tune_wd = 0

        # models
        self.base_model = None
        self.local_model = {}

    def fit_base(self, train_set, valid_set=None, valid_train_set=None):
        if self.few_shot_learning == "local_only":
            self.base_model = None
        elif self.few_shot_learning == "MAML":
            raise NotImplemented
            self.fine_tune_lr = self.meta_inner_lr
            self.fine_tune_num_steps = self.meta_test_num_steps * 2
            self.base_model = self._fit_a_MAML_model(train_set, valid_set, valid_train_set)
        else:
            self.base_model = self._fit_a_model(train_set, valid_set, valid_train_set)

    def fit_local(self, train_set, valid_set=None):
        if self.few_shot_learning == "base_only":
            return
        elif self.few_shot_learning == "local_only_mix_task":
            local_model = self._fit_a_model(train_set, valid_set)
            for task in train_set.tasks():
                self.local_model[task] = local_model
        elif self.few_shot_learning == "local_only_per_task":
            for task in train_set.tasks():
                task_train_set = train_set.extract_subset([task])
                local_model = self._fit_a_model(task_train_set, valid_set)
                self.local_model[task] = local_model
        elif self.few_shot_learning == "plus_mix_task":
            self.net_params["hidden_dim"] = 128
            self.loss_type = 'rmse'
            self.loss_func = torch.nn.MSELoss()
            self.net_params['add_sigmoid'] = True
            base_preds = self._predict_a_dataset(self.base_model, train_set)
            diff_train_set = Dataset()
            for task in train_set.tasks():
                diff_train_set.load_task_data(
                    task,
                    train_set.features[task],
                    train_set.throughputs[task] - base_preds[task]
                )

            if valid_set:
                base_preds = self._predict_a_dataset(self.base_model, valid_set)
                diff_valid_set = Dataset()
                for task in valid_set.tasks():
                    diff_valid_set.load_task_data(
                        task,
                        valid_set.features[task],
                        valid_set.throughputs[task] - base_preds[task]
                    )
            else:
                diff_valid_set = None

            diff_model = self._fit_a_model(diff_train_set, diff_valid_set)

            for task in train_set.tasks():
                self.local_model[task] = diff_model
        elif self.few_shot_learning == "plus_per_task":
            base_preds = self._predict_a_dataset(self.base_model, train_set)
            for task in train_set.tasks():
                diff_train_set = Dataset()
                diff_train_set.load_task_data(
                    task,
                    train_set.features[task],
                    train_set.throughputs[task] - base_preds[task]
                )
                diff_model = self._fit_a_model(diff_train_set, valid_set)
                self.local_model[task] = diff_model
        elif self.few_shot_learning == "fine_tune_mix_task":
            self.base_model = self._fine_tune_a_model(self.base_model, train_set, valid_set)
        else:
            raise ValueError("Invalid few-shot learning method: " + self.few_shot_learning)

    def predict(self, dataset):
        if self.few_shot_learning in ["base_only", "fine_tune_mix_task", "fine_tune_per_task", "MAML"]:
            return self._predict_a_dataset(self.base_model, dataset)
        elif self.few_shot_learning in ["local_only_mix_task", "local_only_per_task"]:
            ret = {}
            for task in dataset.tasks():
                local_preds = self._predict_a_task(self.local_model[task], task, dataset.features[task])
                ret[task] = local_preds
            return ret
        elif self.few_shot_learning in ["plus_mix_task", "plus_per_task"]:
            base_preds = self._predict_a_dataset(self.base_model, dataset)
            ret = {}
            for task in dataset.tasks():
                if task not in self.local_model and self.few_shot_learning == "plus_mix_task":
                    self.local_model[task] = list(self.local_model.values())[0]
                local_preds = self._predict_a_task(self.local_model[task], task, dataset.features[task])
                ret[task] = base_preds[task] + local_preds
            return ret
        else:
            raise ValueError("Invalid few show learing: " + self.few_shot_learning)

    def _fit_a_model(self, train_set, valid_set=None, valid_train_set=None, n_epoch=None):
        print("=" * 60 + "\nFit a net. Train size: %d" % len(train_set))

        for task in train_set.tasks():
            self.register_new_task(task)

        train_loader = SegmentDataLoader(
            train_set, self.batch_size, self.device, self.use_workload_embedding, self.use_target_embedding,
            self.target_id_dict, shuffle=True
        )

        # Normalize features
        if self.fea_norm_vec is None:
            self.fea_norm_vec = train_loader.normalize()
        else:
            train_loader.normalize(self.fea_norm_vec)

        if valid_set:
            for task in valid_set.tasks():
                self.register_new_task(task)
            valid_loader = SegmentDataLoader(valid_set, self.infer_batch_size, self.device, self.use_workload_embedding,
                                             self.use_target_embedding, self.target_id_dict,fea_norm_vec=self.fea_norm_vec)

        n_epoch = n_epoch or self.n_epoch
        early_stop = n_epoch // 6

        net = make_net(self.net_params).to(self.device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.wd
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epoch // 3, gamma=1)

        train_loss = None
        best_epoch = None
        best_train_loss = 1e10
        for epoch in range(n_epoch):
            tic = time.time()

            # train
            net.train()
            for batch, (segment_sizes, features, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.loss_func(net(segment_sizes, features), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                optimizer.step()

                train_loss = moving_average(train_loss, loss.item())
            lr_scheduler.step()

            train_time = time.time() - tic

            if epoch % self.print_per_epoches == 0 or epoch == n_epoch - 1:

                if valid_set and valid_loader:
                    valid_loss = self._validate(net, valid_loader)

                else:
                    valid_loss = 0.0

                if self.loss_type == "rmse":
                    loss_msg = "Train RMSE: %.4f\tValid RMSE: %.4f" % (np.sqrt(train_loss), np.sqrt(valid_loss))
                elif self.loss_type in ["rankNetLoss", "lambdaRankLoss", "listNetLoss"]:
                    loss_msg = "Train Loss: %.4f\tValid Loss: %.4f" % (train_loss, valid_loss)

                print("Epoch: %d\tBatch: %d\t%s\tTrain Speed: %.0f" % (
                    epoch, batch, loss_msg, len(train_loader) / train_time,))

            # Early stop
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_epoch = epoch
            elif epoch - best_epoch >= early_stop:
                print("Early stop. Best epoch: %d" % best_epoch)
                break

            self.save("tmp_mlp.pkl")

        return net

    def register_new_task(self, task):
        target = str(task.target)

        if target not in self.target_id_dict:
            self.target_id_dict[target] = len(self.target_id_dict)


    def _fine_tune_a_model(self, model, train_set, valid_set=None, verbose=1):
        if verbose >= 1:
            print("=" * 60 + "\nFine-tune a net. Train size: %d" % len(train_set))

        # model.freeze_for_fine_tuning()

        train_loader = SegmentDataLoader(
            train_set, self.fine_tune_batch_size or len(train_set),
            self.device, self.use_workload_embedding, fea_norm_vec=self.fea_norm_vec,
        )

        if valid_set:
            valid_loader = SegmentDataLoader(valid_set, self.infer_batch_size,
                                             self.device, self.use_workload_embedding, fea_norm_vec=self.fea_norm_vec)

        tic = time.time()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.fine_tune_lr, weight_decay=self.fine_tune_wd)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.fine_tune_lr, weight_decay=self.wd)
        for step in range(self.fine_tune_num_steps):
            # train
            model.train()
            train_loss = None
            for batch, (segment_sizes, features, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.loss_func(model(segment_sizes, features), labels)
                loss.backward()
                optimizer.step()

                train_loss = moving_average(train_loss, loss.item())

            if verbose >= 1:
                if valid_set:
                    valid_loss = self._validate(model, valid_loader)
                else:
                    valid_loss = 0

                if self.loss_type == "rmse":
                    loss_msg = "Train RMSE: %.4f\tValid RMSE: %.4f" % (np.sqrt(train_loss), np.sqrt(valid_loss))
                elif self.loss_type in ["rankNetLoss", "lambdaRankLoss", "listNetLoss"]:
                    loss_msg = "Train Loss: %.4f\tValid Loss: %.4f" % (train_loss, valid_loss)
                print("Fine-tune step: %d\t%s\tTime: %.1f" % (step, loss_msg, time.time() - tic,))

        return model

    def _validate(self, model, valid_loader):
        model.eval()
        valid_losses = []

        for segment_sizes, features, labels in valid_loader:
            preds = model(segment_sizes, features)
            valid_losses.append(self.loss_func(preds, labels).item())

        return np.mean(valid_losses)

    def _predict_a_dataset(self, model, dataset):
        ret = {}
        for task, features in dataset.features.items():
            ret[task] = self._predict_a_task(model, task, features)
        return ret

    def _predict_a_task(self, model, task, features):
        if model is None:
            return np.zeros(len(features), dtype=np.float32)

        tmp_set = Dataset.create_one_task(task, features, np.zeros((len(features),)))

        preds = []
        for segment_sizes, features, labels in SegmentDataLoader(
                tmp_set, self.infer_batch_size, self.device,
                self.use_workload_embedding, self.use_target_embedding, self.target_id_dict, fea_norm_vec=self.fea_norm_vec,
        ):
            preds.append(model(segment_sizes, features))
        return torch.cat(preds).detach().cpu().numpy()

    def _fit_a_MAML_model(self, train_set, valid_set=None, valid_train_set=None):
        print("=" * 60 + "\nFit a MAML net. Train size: %d" % len(train_set))
        batch_size_tasks = self.meta_batch_size_tasks
        batch_size_per_task = self.meta_batch_size_per_task
        few_shot_number = self.few_shot_number

        print_per_batches = 20
        n_batches = 3000
        early_stop = 200

        # Compute normalization vector over the whole dataset
        if self.fea_norm_vec is None:
            all_train_loader = SegmentDataLoader(
                train_set, self.batch_size, self.device, self.use_workload_embedding,
            )
            self.fea_norm_vec = all_train_loader.normalize()
            del all_train_loader

        # Build dataloaders
        train_loaders = {}
        for task in train_set.feature_data:
            task_dataset = train_set.extract_subset(task)
            train_loaders[task] = SegmentDataLoader(
                task_dataset, None, self.device, self.use_workload_embedding,
                fea_norm_vec=self.fea_norm_vec, shuffle=True,
            )

        # Make network
        net = make_net(self.net_params).to(self.device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.meta_outer_lr, weight_decay=self.wd
        )

        # Training
        avg_outer_loss = None
        avg_inner_loss = None
        task_list = list(train_set.tasks())
        best_batch = None
        best_train_loss = 1e10
        for batch in range(n_batches):
            tasks = random.choices(task_list, k=batch_size_tasks)
            net.train()
            outer_loss = torch.tensor(0.0, device=self.device)
            # outer loss
            for task in tasks:
                train_loader = train_loaders[task]

                train_segment_sizes, train_features, train_labels = train_loader.sample_batch(
                    few_shot_number
                )
                test_segment_sizes, test_features, test_labels = train_loader.sample_batch(
                    batch_size_per_task
                )

                # inner loss
                params = OrderedDict(net.meta_named_parameters())
                for _ in range(self.meta_test_num_steps):
                    inner_loss = self.loss_func(
                        net(train_segment_sizes, train_features, params=params), train_labels
                    )
                    params = gradient_update_parameters(
                        net,
                        inner_loss,
                        params=params,
                        step_size=self.meta_inner_lr,
                        first_order=False,
                    )
                    avg_inner_loss = moving_average(avg_inner_loss, inner_loss.item())

                # acculate gradient for meta-update
                outer_loss += self.loss_func(
                    net(test_segment_sizes, test_features, params=params), test_labels
                )

            optimizer.zero_grad()
            outer_loss /= len(tasks)
            outer_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
            optimizer.step()

            avg_outer_loss = moving_average(avg_outer_loss, outer_loss.item())

            if batch % print_per_batches == 0 or batch == n_batches - 1:
                # validate
                valid_loss = self._validate(net, valid_set, valid_train_set, verbose=0)
                print(
                    "Task Batch: %d\tOuter RMSE: %.4f\tInner RMSE: %.4f\tValid RMSE: %.4f"
                    % (
                        batch,
                        np.sqrt(avg_outer_loss),
                        np.sqrt(avg_inner_loss),
                        np.sqrt(valid_loss),
                    )
                )

            # Early stop
            if avg_outer_loss < best_train_loss:
                best_train_loss = avg_outer_loss
                best_batch = batch
            elif batch - best_batch >= early_stop:
                print("Early stop. Best batch: %d" % best_batch)
                break

        return net

    def load(self, filename):
        if self.device == 'cpu':
            self.base_model, self.local_model, self.few_shot_learning, self.fea_norm_vec = \
                CPU_Unpickler(open(filename, 'rb')).load()
        else:
            self.base_model, self.local_model, self.few_shot_learning, self.fea_norm_vec = \
                pickle.load(open(filename, 'rb'))
            self.base_model = self.base_model.cuda() if self.base_model else None 
            self.local_model = self.local_model.cuda() if self.local_model else None

    def save(self, filename):
        base_model = self.base_model.cpu() if self.base_model else None 
        local_model = self.local_model.cpu() if self.local_model else None
        pickle.dump((base_model, local_model, self.few_shot_learning, self.fea_norm_vec),
                    open(filename, 'wb'))
        self.base_model = self.base_model.to(self.device) if self.base_model else None 
        self.local_model = self.local_model.to(self.device) if self.local_model else None

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class MLPModel(PythonBasedModel):
    """The wrapper of MLPModelInternal. So we can use it in end-to-end search."""

    def __init__(self, few_shot_learning="base_only", disable_update=False):
        super().__init__()

        self.disable_update = disable_update
        self.model = MLPModelInternal(few_shot_learning=few_shot_learning)
        self.dataset = Dataset()

    def update(self, inputs, results):
        if self.disable_update or len(inputs) <= 0:
            return
        tic = time.time()
        self.dataset.update_from_measure_pairs(inputs, results)
        self.model.fit_base(self.dataset)
        logger.info("MLPModel Training time: %.2f s", time.time() - tic)

    def predict(self, task, states):
        features = get_per_store_features_from_states(states, task)
        if self.model is not None:
            learning_task = LearningTask(task.workload_key, str(task.target))
            eval_dataset = Dataset.create_one_task(learning_task, features, None)
            ret = self.model.predict(eval_dataset)[learning_task]
        else:
            ret = np.random.uniform(0, 1, (len(states),))

        # Predict 0 for invalid states that failed to be lowered.
        for idx, feature in enumerate(features):
            if feature.min() == feature.max() == 0:
                ret[idx] = float('-inf')

        return ret

    def update_from_file(self, file_name, n_lines=None):
        inputs, results = RecordReader(file_name).read_lines(n_lines)
        logger.info("MLPModel: Loaded %s measurement records from %s", len(inputs), file_name)
        self.update(inputs, results)

    def save(self, file_name: str):
        self.model.save(file_name)

    def load(self, file_name: str):
        if self.model is None:
            self.model = MLPModelInternal()
        self.model.load(file_name)
        self.num_warmup_sample = -1


def vec_to_pairwise_prob(vec):
    s_ij = vec - vec.unsqueeze(1)
    p_ij = 1 / (torch.exp(s_ij) + 1)
    return torch.triu(p_ij, diagonal=1)


class RankNetLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        preds_prob = vec_to_pairwise_prob(preds)
        labels_prob = torch.triu((labels.unsqueeze(1) > labels).float(), diagonal=1)
        return torch.nn.functional.binary_cross_entropy(preds_prob, labels_prob)


class LambdaRankLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def lamdbaRank_scheme(self, G, D, *args):
        return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
            G[:, :, None] - G[:, None, :])

    def forward(self, preds, labels, k=None, eps=1e-10, mu=10., sigma=1., device=None):
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda:0'
            else:
                device = 'cpu'
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        weights = self.lamdbaRank_scheme(G, D, mu, true_sorted_by_preds)

        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs[torch.isnan(scores_diffs)] = 0.
        weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        losses = torch.log2(weighted_probas)
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        loss = -torch.sum(masked_losses)
        return loss


class ListNetLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels, eps=1e-10):
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + eps
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))
