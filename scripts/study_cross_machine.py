"""Compare the throughputs of the same program on two different machines"""
import json
import os
import random

import numpy as np

import tvm
from tvm.auto_scheduler import RecordReader
from tvm.auto_scheduler.cost_model.metric import (
    metric_rmse,
    metric_r_squared,
    metric_pairwise_comp_accuracy,
    metric_top_k_recall,
    metric_peak_score,
)

from common import MEASURE_RECORD_FOLDER, load_and_register_tasks

def get_normalized_throughput(filename):
    costs_0 = []
    for i, line in enumerate(open(filename)):
        item = json.loads(line)
        cost = np.median(item['r'][0])
        costs_0.append(cost)
    costs_0 = np.array(costs_0)
    throughputs_0 = np.min(costs_0) / costs_0

    return throughputs_0


if __name__ == "__main__":
    target_0 = tvm.target.Target('llvm -mcpu=core-avx2 -model=e5-2666')
    target_1 = tvm.target.Target('llvm -mcpu=skylake-avx512 -model=platinum-8272')

    max_lines = 2

    print("Load task...")
    tasks = load_and_register_tasks()
    #random.seed(0)
    #random.shuffle(tasks)

    for i, task in enumerate(tasks):
        task_key = (task.workload_key, str(task.target.kind))

        file_0 = f"{MEASURE_RECORD_FOLDER}/{target_0.model}/{task_key}.json"
        file_1 = f"{MEASURE_RECORD_FOLDER}/{target_1.model}/{task_key}.json"

        if not os.path.exists(file_0) or not os.path.exists(file_1):
            continue

        throughputs_0 = get_normalized_throughput(file_0)
        throughputs_1 = get_normalized_throughput(file_1)

        if len(throughputs_0) < 1024:
            continue

        rmse = metric_rmse(throughputs_0, throughputs_1)
        r_squared = metric_r_squared(throughputs_0, throughputs_1)
        pair_acc = metric_pairwise_comp_accuracy(throughputs_0, throughputs_1)
        peak_score_1 = metric_peak_score(throughputs_0, throughputs_1, 1)
        peak_score_5 = metric_peak_score(throughputs_0, throughputs_1, 5)

        print("=" * 20)
        print(f"Task: {i}")
        print(f"#programs: {len(throughputs_0)}")
        print(f"flops: {task.compute_dag.flop_ct}")
        print(f"rmse: {rmse}")
        #print(f"R^2: {r_squared}")
        #print(f"pairwise accuracy: {pair_acc}")
        print(f"peak score@1: {peak_score_1}")
        print(f"peak score@5: {peak_score_5}")
        print(task.compute_dag, flush=True)

