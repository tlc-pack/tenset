"""Evaluate a cost model on a network with dataset simulator"""
import argparse
import os
import pickle

import numpy as np

import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler.utils import to_str_round
from tvm.auto_scheduler.dataset import LearningTask
from tvm.auto_scheduler.cost_model.xgb_model import XGBModelInternal
from tvm.auto_scheduler.cost_model.mlp_model import MLPModelInternal

from common import get_task_info_filename, get_measure_record_filename
from train_model import evaluate_model


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


def eval_cost_model_on_network(model, network_key, target, top_ks):
    # Read tasks of the network
    target = tvm.target.Target(target)
    task_info_filename = get_task_info_filename(network_key, target)
    tasks, task_weights = pickle.load(open(task_info_filename, "rb"))
    network_task_key2 = (network_key, str(target))

    # Featurizes a dataset 
    dataset_file = f".dataset_cache/{network_task_key2}.network.feature_cache"
    if not os.path.exists(dataset_file):
        # get file names of these tasks
        filenames = []
        for task in tasks:
            filename = get_measure_record_filename(task, target)
            filenames.append(filename)

        # make a dataset
        auto_scheduler.dataset.make_dataset_from_log_file(
            filenames, dataset_file, min_sample_size=0)
    dataset = pickle.load(open(dataset_file, "rb"))

    eval_res = evaluate_model(model, dataset)
    print(to_str_round(eval_res))
    print("===============================================")

    # Make learning tasks and attach weights
    target = dataset.tasks()[0].target
    learning_tasks = [LearningTask(t.workload_key, target) for t in tasks]
    task_dict = {task: weight for task, weight in zip(learning_tasks, task_weights)}

    return eval_cost_model_on_weighted_tasks(model, task_dict, dataset, top_ks)


def eval_cost_model_on_network_combined(model, network_keys, target):
    target = tvm.target.Target(target)
    task = []
    task_weights = []
    dataset = Dataset()
    for network_key in network_keys:
        task_info_filename = get_task_info_filename(network_key, target)
        tmp_tasks, tmp_task_weights = pickle.load(open(task_info_filename, "rb"))
        task += tmp_tasks
        task_weights += tmp_task_weights
        network_task_key2 = (network_key, str(target))

        dataset_file = f".dataset_cache/{network_task_key2}.network.feature_cache"
        if not os.path.exists(dataset_file):
            # get file names of these tasks
            filenames = []
            for task in tasks:
                filename = get_measure_record_filename(task, target)
                filenames.append(filename)

            # make a dataset
            auto_scheduler.dataset.make_dataset_from_log_file(
                filenames, dataset_file, min_sample_size=0)
        tmp_dataset = pickle.load(open(dataset_file, "rb"))
        dataset.update_from_dataset(tmp_dataset)

    eval_res = evaluate_model(model, dataset)
    print(to_str_round(eval_res))


def eval_cost_model_on_log_file(model, log_file, network_key, target, top_ks):
    target = tvm.target.Target(target)
    task_info_filename = get_task_info_filename(network_key, target)
    tasks, task_weights = pickle.load(open(task_info_filename, "rb"))

    dataset_file = "tmp_dataset_file.pkl"
    auto_scheduler.dataset.make_dataset_from_log_file(
        [log_file], dataset_file, min_sample_size=0)
    dataset = pickle.load(open(dataset_file, "rb"))

    target = dataset.tasks()[0].target
    learning_tasks = [LearningTask(t.workload_key, target) for t in tasks]
    task_dict = {task: weight for task, weight in zip(learning_tasks, task_weights)}

    return eval_cost_model_on_weighted_tasks(model, task_dict, dataset, top_ks)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str)
    parser.add_argument("--log-file", type=str)
    parser.add_argument("--log-file-network", type=str)
    parser.add_argument("--combine", type=str)
    args= parser.parse_args()

    model_file = args.model_file
    network_keys = [
        ("resnet_50", [(1, 3, 224,224)]),
        ("mobilenet_v2", [(1, 3, 224,224)]),
        ("resnext_50", [(1, 3, 224,224)]),
        ("bert_base", [(1, 128)]),
        ("bert_tiny", [(1, 128)]),
    ]
    target = "llvm -model=platinum-8272"

    #model = XGBModelInternal()
    model = MLPModelInternal()
    model.load(model_file)

    if args.combine:
        eval_cost_model_on_network_combined(model, network_keys, target, top_ks)


    top_ks = [1, 5]
    top_1_total = []
    top_5_total = []
    for network_key in network_keys:
        latencies, best_latency = eval_cost_model_on_network(model, network_key, target, top_ks)
        for top_k, latency in zip(top_ks, latencies):
            print(f"Network: {network_key}\tTop-{top_k} score: {best_latency / latency}")

        top_1_total.append(best_latency/latencies[0])
        print(f"top 1 score: {best_latency/latencies[0]}")
        top_5_total.append(best_latency / latencies[1])
        print(f"top 5 score: {best_latency / latencies[1]}")

    print(f"average top 1 score is {sum(top_1_total) / len(top_1_total)}")
    print(f"average top 5 score is {sum(top_5_total) / len(top_5_total)}")

    if args.log_file:
        if args.log_file_network == "resnet_50":
            network_key = ("resnet_50", [(1, 3, 224,224)])
        latencies, best_latency = eval_cost_model_on_log_file(model, args.log_file, network_key, target, top_ks)
        for top_k, latency in zip(top_ks, latencies):
            print(f"Log file\tTop-{top_k} score: {best_latency / latency}")



