"""GraphDataset management"""
from collections import namedtuple, OrderedDict, defaultdict
import os
import pickle
from typing import List, Tuple
from tqdm import tqdm 
import numpy as np

from .measure_record import RecordReader
from .measure import MeasureInput, MeasureResult
from .graph_feature import get_graph_from_measure_pairs

LearningTask = namedtuple("LearningTask", ['workload_key', 'target'])


def input_to_learning_task(inp: MeasureInput):
    return LearningTask(inp.task.workload_key, str(inp.task.target))

DATASET_FORMAT_VERSION = 0.1

class GraphDataset:
    def __init__(self):
        self.raw_files = []

        self.feature_data = {}     # Dict[LearningTask -> Tuple[feature, normalized_throughputs]
        self.measure_records = {}  # Dict[LearningTask -> Tuple[List[MeasureInput], List[MeasureResult]]

        self.format_version = DATASET_FORMAT_VERSION

    def load_raw_files(self, files: List[str]):
        print("Load raw files...")
        for input_file in tqdm(files):
            for inp, res in RecordReader(input_file):
                task = input_to_learning_task(inp)
                if task not in self.measure_records:
                    self.measure_records[task] = [[], []]
                self.measure_records[task][0].append(inp)
                self.measure_records[task][1].append(res)
        self.raw_files.extend(files)

    def featurize(self):
        for task, (inputs, results) in tqdm(self.measure_records.items()):
            fea_throughput_pairs, task_ids = get_graph_from_measure_pairs(inputs, results)
            assert not np.any(task_ids)  # all task ids should be zero
            self.feature_data[task] = fea_throughput_pairs

    def load_task_feature_data(self, task: LearningTask, fea_throughput_pairs):
        self.feature_data[task] = fea_throughput_pairs

    def random_split_within_task(self, train_set_ratio: float) -> Tuple["Dataset", "Dataset"]:
        train_set = GraphDataset()
        test_set = GraphDataset()

        for task, pairs in self.feature_data.items():
            perm = np.random.permutation(len(pairs))
            split = int(train_set_ratio * len(pairs))
            train_pairs = [pairs[i] for i in perm[:split]]
            test_pairs = [pairs[i] for i in perm[split:]]

            train_set.load_task_feature_data(task, train_pairs)
            test_set.load_task_feature_data(task, test_pairs)
        return train_set, test_set

    def tasks(self) -> List[LearningTask]:
        if self.feature_data:
            return list(self.feature_data.keys())
        else:
            return list(self.measure_records.keys())

    def load(self, input_file):
        self.raw_files, self.feature_data, self.format_version =\
                pickle.load(open(input_file, 'rb'))

    def save(self, output_file):
        saved_tuple = (self.raw_files, self.feature_data, self.format_version)
        pickle.dump(saved_tuple, open(output_file, 'wb'))

    def __len__(self, ):
        return sum(len(pairs) for pairs in self.feature_data.values())

def make_dataset_from_log_file(log_files, out_file, min_sample_size, verbose=1):
    """Make a dataset file from raw log files"""
    from tqdm import tqdm

    cache_folder = ".dataset_cache"
    os.makedirs(cache_folder, exist_ok=True)

    dataset = GraphDataset()
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
                    get_graph_from_measure_pairs(inputs, results)

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

