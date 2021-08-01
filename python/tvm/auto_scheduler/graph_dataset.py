"""GraphDataset management"""
from typing import List, Tuple
from collections import namedtuple, OrderedDict
import pickle

from tqdm import tqdm
import numpy as np

from tvm.auto_scheduler.measure_record import RecordReader
from .measure import MeasureInput, MeasureResult
from .feature import get_graph_from_measure_pairs

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

