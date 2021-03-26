"""Evaluate a cost model on a network with dataset simulator"""
import argparse
import os
import pickle

import numpy as np

import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler.dataset import LearningTask
from tvm.auto_scheduler.cost_model.xgb_model import XGBModelInternal

from common import MEASURE_RECORD_FOLDER, NETWORK_INFO_FOLDER


def eval_cost_model_on_weighted_tasks(model, eval_task_dict, eval_dataset, top_ks):
    """Evaluate a cost model on weighted tasks"""
    preds_dict = model.predict(eval_dataset)

    best_latency = 0
    latencies = [0] * len(top_ks)
    for task, weight in eval_task_dict.items():
        if task not in eval_dataset.throughputs:
            print(f"Warning: cannot find {task.workload_key} in the eval_dataset. Skipped.")
            continue

        preds = preds_dict[task]
        labels, min_latency = eval_dataset.throughputs[task], eval_dataset.min_latency[task]

        real_values = labels[np.argsort(-preds)]
        real_latency = min_latency / np.maximum(real_values, 1e-5)

        for i, top_k in enumerate(top_ks):
            latencies[i] += np.min(real_latency[:top_k]) * weight
        best_latency += min_latency * weight

    return latencies, best_latency


def eval_cost_model_on_network(model, network, target, top_ks):
    # Read tasks of the network
    target = tvm.target.Target(target)
    network_task_key = (network_key, str(target.kind))
    task_info_filename = f"dataset/network_info/{network_task_key}.task.pkl"
    tasks, task_weights = pickle.load(open(task_info_filename, "rb"))
    network_task_key2 = (network_key, str(target))

    dataset_file = f".dataset_cache/{network_task_key2}.network.pkl"
    if not os.path.exists(dataset_file):
        # get file names of these tasks
        filenames = []
        for task in tasks:
            task_key = (task.workload_key, str(task.target.kind))
            filename = f"dataset/measure_records/{target.model}/{task_key}.json"
            filenames.append(filename)

        # make a dataset
        auto_scheduler.dataset.make_dataset_from_log_file(
            filenames, dataset_file, min_sample_size=0)

    dataset = pickle.load(open(dataset_file, "rb"))
    target = dataset.tasks()[0].target

    learning_tasks = [LearningTask(t.workload_key, target) for t in tasks]

    task_dict = {task: weight for task, weight in zip(learning_tasks, task_weights)}
    return eval_cost_model_on_weighted_tasks(model, task_dict, dataset, top_ks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str)
    args= parser.parse_args()

    model_file = args.model_file
    network_keys = [
        ("resnet_50", [(1, 3, 224,224)]),
        ("mobilenet_v2", [(1, 3, 224,224)]),
        ("mobilenet_v3", [(1, 3, 224,224)]),
        ("bert_base", [(1, 128)]),
    ]
    target = "llvm -model=e5-2666"

    model = XGBModelInternal()
    model.load(model_file)

    top_ks = [1, 5]
    for network_key in network_keys:
        latencies, best_latency = eval_cost_model_on_network(model, network_key, target, top_ks)
        for top_k, latency in zip(top_ks, latencies):
            print(f"Network: {network_key}\tTop-{top_k} score: {best_latency / latency}")

