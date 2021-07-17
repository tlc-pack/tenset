"""Dataset management"""
from collections import namedtuple, OrderedDict, defaultdict
import os
import pickle
from typing import List, Tuple

import numpy as np

from .measure_record import RecordReader
from .measure import MeasureInput, MeasureResult
from .feature import get_per_store_features_from_measure_pairs

LearningTask = namedtuple("LearningTask", ['workload_key', 'target'])


def input_to_learning_task(inp: MeasureInput):
    return LearningTask(inp.task.workload_key, str(inp.task.target))


DATASET_FORMAT_VERSION = 0.1


class Dataset:
    def __init__(self):
        self.raw_files = None

        self.features = OrderedDict()      # Dict[LearningTask -> feature]
        self.throughputs = OrderedDict()   # Dict[LearningTask -> normalized_throughputs]
        self.min_latency = {}              # Dict[LearningTask -> min latency]
        self.measure_records = {}          # Dict[LearningTask -> Tuple[List[MeasureInput], List[MeasureResult]]

    @staticmethod
    def create_one_task(task, features, throughputs, min_latency=None):
        """Create a new dataset with one task and its feature and throughput data"""
        ret = Dataset()
        ret.load_task_data(task, features, throughputs, min_latency)
        return ret

    def update_from_measure_pairs(self, inputs: List[MeasureInput], results: List[MeasureResult]):
        new_data = {}  # Dict[LearningTask -> Tuple[List[MeasureInput], List[MeasureResult]]]
        for inp, res in zip(inputs, results):
            learning_task = input_to_learning_task(inp)
            store_tuple = new_data.get(learning_task, None)
            if store_tuple is None:
                store_tuple = ([], [])
                new_data[learning_task] = store_tuple
            store_tuple[0].append(inp)
            store_tuple[1].append(res)

        for task, (inputs, results) in new_data.items():
            features, normalized_throughputs, task_ids, min_latency =\
                get_per_store_features_from_measure_pairs(inputs, results)

            assert not np.any(task_ids)   # all task ids should be zero
            assert len(min_latency) == 1  # should have only one task

            self.load_task_data(task, features, normalized_throughputs, min_latency[0])

    def update_from_dataset(self, dataset):
        for task in dataset.features:
            if task not in self.features:
                self.features[task] = dataset.features[task]
                self.throughputs[task] = dataset.throughputs[task]
                self.min_latency[task] = dataset.min_latency[task]

    def load_task_data(self, task: LearningTask, features, throughputs, min_latency=None):
        """Load feature and throughputs for one task"""
        if task not in self.features:
            self.features[task] = features
            self.throughputs[task] = throughputs
            self.min_latency[task] = min_latency
        else:
            try:
                self.features[task] = np.concatenate([self.features[task], features])
            except ValueError:
                # Fix the problem of shape mismatch
                new_features = list(self.features[task])
                new_features.extend(features)
                self.features[task] = np.array(new_features, dtype=object)
            assert min_latency is not None
            combined_min_latency = min(self.min_latency[task], min_latency)
            self.throughputs[task] = np.concatenate([
                self.throughputs[task] * (combined_min_latency / self.min_latency[task]),
                throughputs * (combined_min_latency / min_latency)])
            self.min_latency[task] = combined_min_latency

    def random_split_within_task(self,
                                 train_set_ratio: float=None,
                                 train_set_num: int=None,
                                 shuffle_time: bool=False) -> Tuple["Dataset", "Dataset"]:
        """Randomly split the dataset into a training set and a test set.
        Do the split within each task. A measurement record is a basic unit.
        """
        train_set = Dataset()
        test_set = Dataset()

        assert train_set_ratio is not None or train_set_num is not None

        for task in self.features:
            features, throughputs = self.features[task], self.throughputs[task]
            if train_set_num is None:
                split = int(train_set_ratio * len(features))
            else:
                split = train_set_num

            if shuffle_time:
                perm = np.random.permutation(len(features))
                train_indices, test_indices = perm[:split], perm[split:]
            else:
                arange = np.arange(len(features))
                arange = np.flip(arange)
                train_indices, test_indices = arange[:split], arange[split:]

            if len(train_indices):
                train_throughputs = throughputs[train_indices]
                train_min_latency = self.min_latency[task] / np.max(train_throughputs)
                train_set.load_task_data(task, features[train_indices], train_throughputs, train_min_latency)

            if len(test_indices):
                test_throughputs = throughputs[test_indices]
                test_min_latency = self.min_latency[task] / np.max(test_throughputs)
                test_set.load_task_data(task, features[test_indices], test_throughputs, test_min_latency)

        return train_set, test_set

    def random_split_by_task(self, train_set_ratio: float) -> Tuple["Dataset", "Dataset"]:
        """Randomly split the dataset into a training set and a test set.
        Split tasks into two sets. A learning task is a basic unit.
        """
        tasks = list(self.features.keys())
        np.random.shuffle(tasks)

        train_records = int(len(self) * train_set_ratio)

        train_set = Dataset()
        test_set = Dataset()
        ct = 0
        for task in tasks:
            features, throughputs = self.features[task], self.throughputs[task]
            ct += len(features)
            if ct <= train_records:
                train_set.load_task_data(task, features, throughputs, self.min_latency[task])
            else:
                test_set.load_task_data(task, features, throughputs, self.min_latency[task])

        return train_set, test_set

    def random_split_by_target(self, train_set_ratio: float) -> Tuple["Dataset", "Dataset"]:
        """Randomly split the dataset into a training set and a test set.
        Split targets into two sets. A target is a basic unit.
        """
        target_to_task = defaultdict(list)
        for task in self.features.keys():
            target_to_task[str(task.target)].append(task)
        targets = list(target_to_task.keys())
        targets = list(reversed(targets))
        #np.random.shuffle(targets)

        train_records = int(len(self) * train_set_ratio)

        train_set = Dataset()
        test_set = Dataset()
        ct = 0
        for target in targets:
            tmp_adder = 0
            for task in target_to_task[target]:
                features, normalized_throughputs = self.features[task], self.throughputs[task]
                tmp_adder += len(features)
                if ct <= train_records:
                    train_set.load_task_data(task, features, normalized_throughputs)
                else:
                    test_set.load_task_data(task, features, normalized_throughputs)
            ct += tmp_adder

        return train_set, test_set

    def tasks(self) -> List[LearningTask]:
        """Get all tasks"""
        if self.features:
            return list(self.features.keys())
        else:
            return list(self.measure_records.keys())

    def targets(self) -> List[str]:
        """Get all targest"""
        ret = set()
        for t in self.tasks():
            ret.add(t.target)
        return list(ret)

    def extract_subset(self, tasks: List[LearningTask]) -> "Dataset":
        """Extract a subset containing given tasks"""
        ret = Dataset()
        for task in tasks:
            if not (task in self.features):
                continue
            ret.load_task_data(task, self.features[task], self.throughputs[task], self.min_latency[task])
        return ret

    def __getstate__(self):
        return self.raw_files, self.features, self.throughputs, self.min_latency, DATASET_FORMAT_VERSION

    def __setstate__(self, value):
        self.raw_files, self.features, self.throughputs, self.min_latency, format_version = value

    def __len__(self, ):
        return sum(len(x) for x in self.throughputs.values())


def make_dataset_from_log_file(log_files, out_file, min_sample_size, verbose=1):
    """Make a dataset file from raw log files"""
    from tqdm import tqdm

    cache_folder = ".dataset_cache"
    os.makedirs(cache_folder, exist_ok=True)

    dataset = Dataset()
    dataset.raw_files = log_files
    for filename in tqdm(log_files):
        assert os.path.exists(filename), f"{filename} does not exist."

        cache_file = f"{cache_folder}/{filename.replace('/', '_')}.feature_cache"
        if os.path.exists(cache_file):
            # Load feature from the cached file
            features, throughputs, min_latency = pickle.load(open(cache_file, "rb"))
        else:
            # Read measure records
            measure_records = {}
            for inp, res in RecordReader(filename):
                task = input_to_learning_task(inp)
                if task not in measure_records:
                    measure_records[task] = [[], []]
                measure_records[task][0].append(inp)
                measure_records[task][1].append(res)

            # Featurize
            features = {}
            throughputs = {}
            min_latency = {}
            for task, (inputs, results) in measure_records.items():
                features_, normalized_throughputs, task_ids, min_latency_ =\
                    get_per_store_features_from_measure_pairs(inputs, results)

                assert not np.any(task_ids)   # all task ids should be zero
                if len(min_latency_) == 0:
                    # no valid records
                    continue
                else:
                    # should have only one task
                    assert len(min_latency_) == 1, f"len = {len(min_latency)} in {filename}"

                features[task] = features_
                throughputs[task] = normalized_throughputs
                min_latency[task] = min_latency_[0]
            pickle.dump((features, throughputs, min_latency), open(cache_file, "wb"))

        for task in features:
            dataset.load_task_data(task, features[task], throughputs[task], min_latency[task])

    # Delete task with too few samples
    to_delete = []
    for i, (task, feature) in enumerate(dataset.features.items()):
        if verbose >= 0:
            print("No: %d\tTask: %s\tSize: %d" % (i, task, len(feature)))
        if len(feature) < min_sample_size:
            if verbose >= 0:
                print("Deleted")
            to_delete.append(task)
    for task in to_delete:
        del dataset.features[task]
        del dataset.throughputs[task]
        del dataset.min_latency[task]

    # Save to disk
    pickle.dump(dataset, open(out_file, "wb"))

    if verbose >= 0:
        print("A dataset file is saved to %s" % out_file)

