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

import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import numpy as np
from .sparsemax import Sparsemax, Entmax15

class AttentiveTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):
        """
        Initialize an attention transformer.
        Parameters
        ----------
        input_dim : int
            Input size
        output_dim : int
            Output_size
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(AttentiveTransformer, self).__init__()
        self.fc = Linear(input_dim, output_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, output_dim)
        self.bn = GBN(
            output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

        if mask_type == "sparsemax":
            # Sparsemax
            self.selector = Sparsemax(dim=-1)
        elif mask_type == "entmax":
            # Entmax
            self.selector = Entmax15(dim=-1)
        else:
            raise NotImplementedError(
                "Please choose either sparsemax" + "or entmax as masktype"
            )

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.selector(x)
        return x


class FeatTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        shared_layers,
        n_glu_independent,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(FeatTransformer, self).__init__()
        """
        Initialize a feature transformer.
        Parameters
        ----------
        input_dim : int
            Input size
        output_dim : int
            Output_size
        shared_layers : torch.nn.ModuleList
            The shared block that should be common to every step
        n_glu_independent : int
            Number of independent GLU layers
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization within GLU block(s)
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        """

        params = {
            "n_glu": n_glu_independent,
            "virtual_batch_size": virtual_batch_size,
            "momentum": momentum,
        }

        if shared_layers is None:
            # no shared layers
            self.shared = torch.nn.Identity()
            is_first = True
        else:
            self.shared = GLU_Block(
                input_dim,
                output_dim,
                first=True,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
            )
            is_first = False

        if n_glu_independent == 0:
            # no independent layers
            self.specifics = torch.nn.Identity()
        else:
            spec_input_dim = input_dim if is_first else output_dim
            self.specifics = GLU_Block(
                spec_input_dim, output_dim, first=is_first, **params
            )

    def forward(self, x):
        x = self.shared(x)
        x = self.specifics(x)
        return x


class GLU_Block(torch.nn.Module):
    """
    Independent GLU block, specific to each step
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        n_glu=2,
        first=False,
        shared_layers=None,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = torch.nn.ModuleList()

        params = {"virtual_batch_size": virtual_batch_size, "momentum": momentum}

        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLU_Layer(input_dim, output_dim, fc=fc, **params))
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU_Layer(output_dim, output_dim, fc=fc, **params))

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x


class GLU_Layer(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, fc=None, virtual_batch_size=128, momentum=0.02
    ):
        super(GLU_Layer, self).__init__()

        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)

        self.bn = GBN(
            2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.output_dim], torch.sigmoid(x[:, self.output_dim :]))
        return out

def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


class TabNetEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):
        """
        Defines main part of the TabNet network without the embedding layers.
        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(TabNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = BatchNorm1d(self.input_dim, momentum=0.01)

        if self.n_shared > 0:
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(
                        Linear(self.input_dim, 2 * (n_d + n_a), bias=False)
                    )
                else:
                    shared_feat_transform.append(
                        Linear(n_d + n_a, 2 * (n_d + n_a), bias=False)
                    )

        else:
            shared_feat_transform = None

        self.initial_splitter = FeatTransformer(
            self.input_dim,
            n_d + n_a,
            shared_feat_transform,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=momentum,
        )

        self.feat_transformers = torch.nn.ModuleList()
        self.att_transformers = torch.nn.ModuleList()

        for step in range(n_steps):
            transformer = FeatTransformer(
                self.input_dim,
                n_d + n_a,
                shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            attention = AttentiveTransformer(
                n_a,
                self.input_dim,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
                mask_type=self.mask_type,
            )
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)

    def forward(self, x, prior=None):
        x = self.initial_bn(x)

        if prior is None:
            prior = torch.ones(x.shape).to(x.device)

        M_loss = 0
        att = self.initial_splitter(x)[:, self.n_d :]

        steps_output = []
        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            M_loss += torch.mean(
                torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1)
            )
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, : self.n_d])
            steps_output.append(d)
            # update attention
            att = out[:, self.n_d :]

        M_loss /= self.n_steps
        return steps_output, M_loss

    def forward_masks(self, x):
        x = self.initial_bn(x)

        prior = torch.ones(x.shape).to(x.device)
        M_explain = torch.zeros(x.shape).to(x.device)
        att = self.initial_splitter(x)[:, self.n_d :]
        masks = {}

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            masks[step] = M
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, : self.n_d])
            # explain
            step_importance = torch.sum(d, dim=1)
            M_explain += torch.mul(M, step_importance.unsqueeze(dim=1))
            # update attention
            att = out[:, self.n_d :]

        return M_explain, masks


class TabNetDecoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        n_d=8,
        n_steps=3,
        n_independent=1,
        n_shared=1,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        """
        Defines main part of the TabNet network without the embedding layers.
        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 1)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 1)
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        """
        super(TabNetDecoder, self).__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_steps = n_steps
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size

        self.feat_transformers = torch.nn.ModuleList()

        if self.n_shared > 0:
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(Linear(n_d, 2 * n_d, bias=False))
                else:
                    shared_feat_transform.append(Linear(n_d, 2 * n_d, bias=False))

        else:
            shared_feat_transform = None

        for step in range(n_steps):
            transformer = FeatTransformer(
                n_d,
                n_d,
                shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            self.feat_transformers.append(transformer)

        self.reconstruction_layer = Linear(n_d, self.input_dim, bias=False)
        initialize_non_glu(self.reconstruction_layer, n_d, self.input_dim)

    def forward(self, steps_output):
        res = 0
        for step_nb, step_output in enumerate(steps_output):
            x = self.feat_transformers[step_nb](step_output)
            res = torch.add(res, x)
        res = self.reconstruction_layer(res)
        return res

class TabNetNoEmbeddings(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):
        """
        Defines main part of the TabNet network without the embedding layers.
        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(TabNetNoEmbeddings, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = BatchNorm1d(self.input_dim, momentum=0.01)

        self.encoder = TabNetEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
        )

        if self.is_multi_task:
            self.multi_task_mappings = torch.nn.ModuleList()
            for task_dim in output_dim:
                task_mapping = Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
        else:
            self.final_mapping = Linear(n_d, output_dim, bias=False)
            initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, x):
        res = 0
        steps_output, M_loss = self.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        if self.is_multi_task:
            # Result will be in list format
            out = []
            for task_mapping in self.multi_task_mappings:
                out.append(task_mapping(res))
        else:
            out = self.final_mapping(res)
        return out, M_loss

    def forward_masks(self, x):
        return self.encoder.forward_masks(x)


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

        print('building SegmentSumMLPModule.....')
        self.segment_encoder = TabNetNoEmbeddings(in_dim, hidden_dim, 
                                                    n_d=64,
                                                    n_a=64,
                                                    n_steps=7,
                                                    gamma=1.3,
                                                    n_independent=2,
                                                    n_shared=2,
                                                    epsilon=1e-15,
                                                    virtual_batch_size=512,
                                                    momentum=0.02,
                                                    mask_type="entmax",)
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
        )[0]
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

def make_net(params):
    return SegmentSumMLPModule(
            params["in_dim"], params["hidden_dim"], params["out_dim"],
            add_sigmoid=params['add_sigmoid']
        )

def moving_average(average, update):
    if average is None:
        return update
    else:
        return average * 0.95 + update * 0.05


class TabNetModelInternal:
    def __init__(self, use_gpu=True, device=None, few_shot_learning="base_only", use_workload_embedding=True, use_target_embedding=False,
                 loss_type='lambdaRankLoss'):
        print('tabnet')
        if device is None:
            if torch.cuda.device_count() and use_gpu:
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

        self.target_id_dict = {}
        self.loss_type = loss_type
        self.n_epoch = 150
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
        if self.few_shot_learning in ["base_only", "fine_tune_mix_task", "fine_tune_per_task"]:
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


class TabNetModel(PythonBasedModel):
    """The wrapper of TabNetModelInternal. So we can use it in end-to-end search."""

    def __init__(self, few_shot_learning="base_only", disable_update=False):
        super().__init__()

        self.disable_update = disable_update
        self.model = TabNetModelInternal(few_shot_learning=few_shot_learning)
        self.dataset = Dataset()

    def update(self, inputs, results):
        if self.disable_update or len(inputs) <= 0:
            return
        tic = time.time()
        self.dataset.update_from_measure_pairs(inputs, results)
        self.model.fit_base(self.dataset)
        logger.info("TabNetModel Training time: %.2f s", time.time() - tic)

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
        logger.info("TabNetModel: Loaded %s measurement records from %s", len(inputs), file_name)
        self.update(inputs, results)

    def save(self, file_name: str):
        self.model.save(file_name)

    def load(self, file_name: str):
        if self.model is None:
            self.model = TabNetModelInternal()
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