# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name

"""Cost model based on lightgbm"""
from collections import defaultdict
import logging
import multiprocessing
import pickle
import time

import numpy as np

from tvm.autotvm.tuner.metric import max_curve
from tvm.auto_scheduler.compute_dag import ComputeDAG
from tvm.auto_scheduler.dataset import Dataset, LearningTask
from tvm.auto_scheduler.feature import (
    get_per_store_features_from_measure_pairs, get_per_store_features_from_states)
from tvm.auto_scheduler.measure_record import RecordReader
from tvm.auto_scheduler.workload_registry import workload_key_to_tensors
from .cost_model import PythonBasedModel
from ..feature import get_per_store_feature_names

lgbm = None

logger = logging.getLogger("auto_scheduler")


class LGBMDatasetContext:
    """A global context to hold additional attributes of lgbm.Dataset"""

    def __init__(self):
        self.context_dict = defaultdict(dict)

    def get(self, key, matrix, default=None):
        """
        Get an attribute of a lgbm.Dataset
        Parameters
        ----------
        key: str
            The name of the attribute
        matrix: lgbm.Dataset
            The matrix
        default: Optional[Any]
            The default value if the item does not exist
        """
        return self.context_dict[key].get(id(matrix), default)

    def set(self, key, matrix, value):
        """
        Set an attribute for a lgbm.Dataset
        Parameters
        ----------
        key: str
            The name of the attribute
        matrix: lgbm.Dataset
            The matrix
        value: Optional[Any]
            The new value
        """
        self.context_dict[key][id(matrix)] = value


dataset_context = LGBMDatasetContext()


def get_workload_embedding(workload_key):
    tags = ['max', 'min', 'add', 'Conv2dOutput', 'conv2d_winograd', 'DepthwiseConv2d',
            'dense', 'softmax', 'compute(b, i, j)']
    dag_str = str(ComputeDAG(workload_key_to_tensors(workload_key)))
    vec = [0] * len(tags)
    for i, tag in enumerate(tags):
        if tag in dag_str:
            vec[i] = 1
    return vec


class LGBModelInternal:
    """Train a LightGBM model to predict the normalized throughputs of programs.
    Let the normalized throughput be the score of a program (higher is better). We predict
    the (approximate) score of a program = the sum of the scores of all stages in this program.
    i.e. score(P) = score_s0 + score_s1 + ... + score_sn,
    where score_si is the score of Stage i in Program P.
    We extract feature for each stage and let the LightGBM predict the score for each stage.
    We then sum up the predictions as the score of the whole program.
    We use RMSE as the loss function.  i.e. loss(P, y) = 1/2 * (score(P) - y)^2,
    where P is the program and y is the normalized throughput according to
    the ground truth (measurement).
    LightGBM does not support this loss function because `score(P)` is a sum of the prediction
    of several samples, so we implemented a custom loss function and call it pack-sum-rmse.
    It is called "pack-sum" because we combine several samples into a "pack" and sum up
    their predictions.
    """
    def __init__(
        self,
        params=None,
        use_workload_embedding=True,
        use_data_argumentation=False,
        use_gpu=False,
        few_shot_learning="base_only",
        verbose_eval=25,
        seed=None):

        global lgbm
        try:
            if lgbm is None:
                lgbm = __import__("lightgbm")
        except ImportError:
            # add "from Node" to silence
            # "During handling of the above exception, another exception occurred"
            raise ImportError(
                "LightGBM is required for LGBModel. "
                "Please install its python package first. "
            ) from None

        self.plan_size = 1
        self.use_weight = False

        self.use_workload_embedding = use_workload_embedding
        self.use_data_argumentation = use_data_argumentation
        self.few_shot_learning = few_shot_learning
        self.verbose_eval = verbose_eval
        self.workload_embed_dict = dict()

        # lgbm params
        if params is None:
            self.lgbm_params = {
                'boosting_type': 'gbdt',
                'num_leaves': 72,
                'learning_rate': 0.1632095,
                'feature_fraction': 0.84375,
                'bagging_fraction': 0.89435,
                'bagging_freq': 4,
                'verbose': 0,
                'min_sum_hessian_in_leaf': 4,
            }
        else:
            self.lgbm_params = params 
            
        # gpu support
        if use_gpu:
            self.lgbm_params['tree_method'] = 'gpu_hist'

        # models
        self.base_model = None
        self.local_model = {}

    def fit(self, *args, **kwargs):
        return self.fit_base(*args, **kwargs)

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
            diff_train_set = self.make_diff_set(self.base_model, train_set)
            diff_valid_set = self.make_diff_set(self.base_model, valid_set) if valid_set else None
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
        else:
            raise ValueError("Invalid few-shot learning method: " + self.few_shot_learning)

    def predict(self, dataset):
        if self.few_shot_learning == "base_only":
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
                local_preds = self._predict_a_task(self.local_model[task], task, dataset.features[task])
                ret[task] = base_preds[task] + local_preds
            return ret
        else:
            raise ValueError("Invalid few show learing: " + self.few_shot_learning)

    def _fit_a_model(self, train_set, valid_set=None, valid_train_set=None):
        print("Fit a lgbm booster. Train size: %d" % len(train_set))

        for task in train_set.tasks():
            self.register_new_task(task)
        train_set = self.dataset_to_lgbm_dataset(train_set, argumentation=self.use_data_argumentation)

        if valid_set is not None:
            for task in valid_set.tasks():
                self.register_new_task(task)
            test_set = self.dataset_to_lgbm_dataset(valid_set)
            eval_sets = [train_set, test_set]
            eval_names = ['tr','te']
        else:
            eval_sets = [train_set]
            eval_names = ['tr']
        
        # Train a new model
        bst = lgbm.train(
            params=self.lgbm_params,
            train_set=train_set,
            valid_sets=eval_sets,
            valid_names=eval_names,
            num_boost_round=2000,
            fobj=pack_sum_square_error,
            feval=[pack_sum_rmse, pack_sum_average_peak_score(self.plan_size)],
            early_stopping_rounds=100,
            verbose_eval=self.verbose_eval  
        )

        feature_names = list(get_per_store_feature_names()) + ['max', 'min', 'add', 
            'Conv2dOutput', 'conv2d_winograd', 'DepthwiseConv2d',
            'dense', 'softmax', 'compute(b, i, j)']
        feature_importances = bst.feature_importance()
        imp = sorted(list(zip(feature_importances, feature_names)))
        #print("Feature importances: ", imp)

        return bst

    def _predict_a_dataset(self, model, dataset):
        ret = {}
        for task, features in dataset.features.items():
            ret[task] = self._predict_a_task(model, task, features)
        return ret

    def _predict_a_task(self, model, task, features):
        if model is None:
            return np.zeros(len(features), dtype=np.float32)

        # Convert features to dataset
        tmp_set = Dataset.create_one_task(task, features, None)
        dataset = self.dataset_to_lgbm_dataset(tmp_set)

        # Make predictions
        raw_preds = model.predict(dataset.data)
        pack_ids = dataset_context.get("pack_ids", dataset)
        predictions = pack_sum_predict_throughput(raw_preds, pack_ids)
        return predictions

    def register_new_task(self, task):
        pass
        #workload_key = str(task.workload_key)
        #self.workload_embed_dict[workload_key] = get_workload_embedding(workload_key)

    def make_diff_set(self, base_model, dataset):
        base_preds = self._predict_a_dataset(base_model, dataset)
        diff_set = Dataset()
        for task in dataset.tasks():
            diff_set.load_task_data(
                task,
                dataset.features[task],
                dataset.throughputs[task] - base_preds[task]
            )
        return diff_set

    def dataset_to_lgbm_dataset(self, dataset, return_task_order=False, argumentation=False):
        # Process input data to lgbm format
        xs, ys, gids = [], [], []
        task_order = []

        for gid, task in enumerate(dataset.features):
            features, throughputs = dataset.features[task], dataset.throughputs[task]
            task_order.append(task)

            # add task embedding into the feature
            if self.use_workload_embedding:
                if task.workload_key not in self.workload_embed_dict:
                    self.workload_embed_dict[task.workload_key] =\
                        get_workload_embedding(task.workload_key)
                task_embedding = self.workload_embed_dict[task.workload_key]

                extended_features = []
                # append task embedding into feature vectors
                for i in range(len(features)):
                    tmp = np.tile(task_embedding, (len(features[i]), 1))
                    extended_features.append(np.concatenate([features[i], tmp], axis=1))

                xs.extend(extended_features)
            else:
                xs.extend(features)

            if throughputs is None:
                ys.append(np.zeros(len(features), dtype=np.float32))
            else:
                ys.append(throughputs)
            gids.append(np.ones(len(features), dtype=np.int32) * gid)

            if argumentation:
                features = np.copy(features)
                tmp = np.copy(features[:][57 + 18*1:57 + 18*2])
                features[:][57 + 18*1: 57 + 18*2] = features[:][57 + 18*2:57 + 18*3]
                features[:][57 + 18*2: 57 + 18*3] = tmp
                xs.extend(features)
                ys.append(throughputs)
                gids.append(np.ones(len(features), dtype=np.int32) * gid)

        xs = np.array(xs, dtype=object)
        ys = np.concatenate(ys)
        gids = np.concatenate(gids)
        dataset = pack_sum_lgbmdataset(
            xs, ys, gids=gids, weights=np.maximum(ys, 0.1) if self.use_weight else None
        )

        if return_task_order:
            return dataset, task_order
        else:
            return dataset

    def load(self, filename):
        self.base_model, self.local_model, params = \
            pickle.load(open(filename, 'rb'))
        self.few_shot_learning = params['few_shot_learning']
        self.use_workload_embedding = params['use_workload_embedding']

    def save(self, filename):
        params = {
            'few_shot_learning': self.few_shot_learning,
            'use_workload_embedding': self.use_workload_embedding,
        }
        pickle.dump((self.base_model, self.local_model, params),
            open(filename, 'wb'))


class LGBModel(PythonBasedModel):
    """The wrapper of LGBModelInternal. So we can use it in end-to-end search."""
    def __init__(self, few_shot_learning="base_only", verbose_eval=25,
                 num_warmup_sample=100, seed=None, disable_update=False):
        super().__init__()

        self.num_warmup_sample = num_warmup_sample
        self.disable_update = disable_update
        self.model = LGBModelInternal(few_shot_learning=few_shot_learning,
                                      verbose_eval=verbose_eval,
                                      seed=seed)
        self.dataset = Dataset()

    def update(self, inputs, results):
        if self.disable_update or len(inputs) <= 0:
            return
        tic = time.time()
        self.dataset.update_from_measure_pairs(inputs, results)
        self.model.fit_base(self.dataset)
        logger.info("LGBModel Training time: %.2f s", time.time() - tic)

    def predict(self, task, states):
        features = get_per_store_features_from_states(states, task)
        if self.model is not None and len(self.dataset) > self.num_warmup_sample:
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
        """Load measure records from a log file to update the cost model.
        This function can be used to pre-train the cost model with history log files.
        Parameters
        ----------
        file_name: str
            The filename
        n_lines: Optional[int]
            Only load first n lines of the log file
        """
        inputs, results = RecordReader(file_name).read_lines(n_lines)
        logger.info("LGBModel: Loaded %s measurement records from %s", len(inputs), file_name)
        self.update(inputs, results)

    def save(self, file_name: str):
        """Save the model to a file

        Parameters
        ----------
        file_name: str
            The filename
        """
        self.model.save(file_name)

    def load(self, file_name: str):
        """Load the model from a file

        Parameters
        ----------
        file_name: str
            The filename
        """
        if self.model is None:
            self.model = LGBModelInternal()
        self.model.load(file_name)
        self.num_warmup_sample = -1


def feature_to_pack_sum_lgbmdataset(xs):
    """Convert an extracted multi-stage feature vector to a lgbmdataset in pack-sum format
    Parameters
    ----------
    xs: np.ndarray
        The feature vector
    Returns
    -------
    dataset: lgbm.Dataset
        The dataset
    pack_ids: List[int]
        pack ids information
    """
    x_flatten = []
    pack_ids = []

    for ct, x in enumerate(xs):
        for row in x:
            x_flatten.append(row)
            pack_ids.append(ct)

    return lgbm.Dataset(np.array(x_flatten)), pack_ids


def pack_sum_lgbmdataset(xs, ys, gids=None, weights=None):
    """Convert (feature, label) pairs into a lgbm dataset with pack-sum format
    Parameters
    ----------
    xs: np.ndarray
        The feature vector
    ys: np.ndarray
        The normaizlied throughput
    gids: Optional[List[int]]
        Group id (task id)
    weights: Optional[np.ndarray]
        The weight of samples
    Returns
    -------
    dataset: lgbm.Dataset
        The dataset with pack-sum information
    """
    if gids is not None:
        # sort by group
        indices = gids.argsort(kind='stable')
        xs, ys = xs[indices], ys[indices]
        group_sizes = np.bincount(gids)
        if weights is not None:
            weights = weights[indices]
    else:
        # assume it has only one group
        group_sizes = [len(xs)]

    x_flatten = []
    y_flatten = []
    weights_flatten = []
    pack_ids = []

    if weights is not None:
        for ct, (x, y, w) in enumerate(zip(xs, ys, weights)):
            for row in x:
                x_flatten.append(row)
                y_flatten.append(y)
                weights_flatten.append(w)
                pack_ids.append(ct)
    else:
        for ct, (x, y) in enumerate(zip(xs, ys)):
            for row in x:
                x_flatten.append(row)
                y_flatten.append(y)
                pack_ids.append(ct)

    ret = lgbm.Dataset(np.array(x_flatten), y_flatten)
    if weights is not None:
        ret.set_weight(weights_flatten)
    dataset_context.set("pack_ids", ret, np.array(pack_ids))
    dataset_context.set("group_sizes", ret, group_sizes)
    return ret


def pack_sum_predict_throughput(raw_preds, pack_ids):
    """Predict the throughputs for predictions in pack-sum format
    Parameters
    ----------
    raw_preds: np.ndarray
        The raw predictions
    pack_ids: List[int]
        The pack id for predictions
    Returns
    -------
    throughputs: np.ndarray
        The throughput
    """
    sum_pred = np.bincount(pack_ids, weights=raw_preds)
    return sum_pred


def pack_sum_square_error(preds, train_set):
    """Implement square error loss on pack-sum format as
     a custom objective function for lgbmdataset.
    Parameters
    ----------
    preds: np.ndarray
        The predicitons
    train_set: lgbm.Dataset
        The training set
    Returns
    -------
    gradient: np.ndarray
    hessian: np.ndarray
        gradient and hessian according to the lgbmdataset format
    """
    pack_ids = dataset_context.get("pack_ids", train_set)
    weight = train_set.get_weight()

    sum_pred = np.bincount(pack_ids, weights=preds)
    x = sum_pred[pack_ids]
    y = train_set.get_label()
    gradient = x - y
    hessian = np.ones_like(gradient)

    if not weight:
        return gradient, hessian

    return gradient * weight, hessian * weight


def pack_sum_rmse(raw_preds, train_set):
    """Evaluate RMSE (rooted mean square error) in the pack-sum format
    Parameters
    ----------
    raw_preds: np.ndarray
        The raw prediction
    train_set: lgbm.Dataset
        The groud-truth label matrix
    Returns
    -------
    name: str
    score: float
        The name and score of this metric
    """
    pack_ids = dataset_context.get("pack_ids", train_set)
    preds = pack_sum_predict_throughput(raw_preds, pack_ids)
    labels = (np.bincount(pack_ids, weights=train_set.get_label())
              / np.unique(pack_ids, return_counts=True)[1])
    return 'rmse', np.sqrt(np.mean(np.square((preds - labels)))), False


def pack_sum_average_peak_score(N):
    """Return the evaluation function for average-peak-score@N
    Parameters
    ----------
    N: int
        The "N" in "average-peak-score@N"
    Returns
    -------
    The evaluation function
    """

    def feval(preds, labels):
        """Evaluate average-peak-score@N in the pack-sum format
        Parameters
        ----------
        raw_preds: np.ndarray
            The raw prediction
        labels: lgbm.Dataset
            The groud-truth label matrix
        Returns
        -------
        name: str
        score: float
        The name and score of this metric
        """
        group_sizes = dataset_context.get("group_sizes", labels, [len(preds)])
        pack_ids = dataset_context.get("pack_ids", labels)

        preds = pack_sum_predict_throughput(preds, pack_ids)
        labels = (
            np.bincount(pack_ids, weights=labels.get_label())
            / np.unique(pack_ids, return_counts=True)[1]
        )

        scores = []
        offset = 0
        for size in group_sizes:
            preds_group = preds[offset : offset + size]
            labels_group = labels[offset : offset + size]
            offset += size

            trials = np.argsort(preds_group)[::-1][:N]
            trial_scores = labels_group[trials]
            curve = max_curve(trial_scores) / np.max(labels_group)
            scores.append(np.mean(curve))
        return "a-peak@%d" % N, np.mean(scores), True

    return feval
