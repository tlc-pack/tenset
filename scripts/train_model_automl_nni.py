"""Train a cost model with a dataset."""

import argparse
import logging
import pickle
import random
import multiprocessing

import nni

import torch
import numpy as np

import tvm
from tvm.auto_scheduler.utils import to_str_round
from tvm.auto_scheduler.cost_model import RandomModelInternal

from common import load_and_register_tasks, str2bool

from tvm.auto_scheduler.dataset import Dataset, LearningTask
from tvm.auto_scheduler.cost_model.xgb_model import XGBModelInternal
from tvm.auto_scheduler.cost_model.mlp_model import MLPModelInternal
from tvm.auto_scheduler.cost_model.lgbm_model import LGBModelInternal
from tvm.auto_scheduler.cost_model.tabnet_model import TabNetModelInternal
from tvm.auto_scheduler.cost_model.metric import (
    metric_rmse,
    metric_r_squared,
    metric_pairwise_comp_accuracy,
    metric_top_k_recall,
    metric_peak_score,
    random_mix,
)

def get_default_parameters():
    params =  {
            "max_depth": 6,
            "gamma": 0.003,
            "min_child_weight": 2,
            "eta": 0.2,
            "n_gpus": 0,
            "nthread": multiprocessing.cpu_count() // 2,
            "verbosity": 0,
            "seed": 42,
            "disable_default_eval_metric": 1,
        }
    return params

def evaluate_model(model, test_set):
    # make prediction
    prediction = model.predict(test_set)

    # compute weighted average of metrics over all tasks
    tasks = list(test_set.tasks())
    weights = [len(test_set.throughputs[t]) for t in tasks]
    print("Test set sizes:", weights)

    rmse_list = []
    r_sqaured_list = []
    pair_acc_list = []
    peak_score1_list = []
    peak_score5_list = []

    for task in tasks:
        preds = prediction[task]
        labels = test_set.throughputs[task]

        rmse_list.append(np.square(metric_rmse(preds, labels)))
        r_sqaured_list.append(metric_r_squared(preds, labels))
        pair_acc_list.append(metric_pairwise_comp_accuracy(preds, labels))
        peak_score1_list.append(metric_peak_score(preds, labels, 1))
        peak_score5_list.append(metric_peak_score(preds, labels, 5))

    rmse = np.sqrt(np.average(rmse_list, weights=weights))
    r_sqaured = np.average(r_sqaured_list, weights=weights)
    pair_acc = np.average(pair_acc_list, weights=weights)
    peak_score1 = np.average(peak_score1_list, weights=weights)
    peak_score5 = np.average(peak_score5_list, weights=weights)

    eval_res = {
        "RMSE": rmse,
        "R^2": r_sqaured,
        "pairwise comparision accuracy": pair_acc,
        "average peak score@1": peak_score1,
        "average peak score@5": peak_score5,
    }
    return eval_res


def make_model(name, use_gpu=False):
    """Make model according to a name"""
    if name == "xgb":
        return XGBModelInternal(use_gpu=use_gpu)
    elif name == "mlp":
        return MLPModelInternal()
    elif name == 'lgbm':
        return LGBModelInternal(use_gpu=use_gpu)
    elif name == 'tab':
        return TabNetModelInternal(use_gpu=use_gpu)
    elif name == "random":
        return RandomModelInternal()
    else:
        raise ValueError("Invalid model: " + name)
 

def train_zero_shot(PARAMS, dataset, train_ratio, model_names, split_scheme, use_gpu):
    # Split dataset
    if split_scheme == "within_task":
        train_set, test_set = dataset.random_split_within_task(train_ratio)
    elif split_scheme == "by_task":
        train_set, test_set = dataset.random_split_by_task(train_ratio)
    elif split_scheme == "by_target":
        train_set, test_set = dataset.random_split_by_target(train_ratio)
    else:
        raise ValueError("Invalid split scheme: " + split_scheme)

    print("Train set: %d. Task 0 = %s" % (len(train_set), train_set.tasks()[0]))
    if len(test_set) == 0:
        test_set = train_set
    print("Test set:  %d. Task 0 = %s" % (len(test_set), test_set.tasks()[0]))

    # Make models
    names = model_names.split("@")
    models = []
    for name in names:
        model = make_model(name, use_gpu)
        model.xgb_params = PARAMS
        models.append(model)

    eval_results = []
    for name, model in zip(names, models):
        # Train the model
        filename = name + ".pkl"
        model.fit_base(train_set, valid_set=test_set)
        print("Save model to %s" % filename)
        model.save(filename)

        # Evaluate the model
        eval_res = evaluate_model(model, test_set)
        print(name, to_str_round(eval_res))
        eval_results.append(eval_res)

    # Print evaluation results
    for i in range(len(models)):
        print("-" * 60)
        print("Model: %s" % names[i])
        for key, val in eval_results[i].items():
            print("%s: %.4f" % (key, val))
    
    nni.report_final_result(eval_results[0]['average peak score@1'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset.pkl")
    parser.add_argument("--models", type=str, default="xgb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-scheme",
        type=str,
        choices=["by_task", "within_task", "by_target"],
        default="within_task",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--use-gpu", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to use GPU for xgb.")
    args = parser.parse_args()
    print("Arguments: %s" % str(args))

    # Setup random seed and logging
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    logging.basicConfig()
    logging.getLogger("auto_scheduler").setLevel(logging.DEBUG)

    print("Load all tasks...")
    load_and_register_tasks()

    print("Load dataset...")
    dataset = pickle.load(open(args.dataset, "rb"))

    RECEIVED_PARAMS = nni.get_next_parameter()
    PARAMS = get_default_parameters()
    PARAMS.update(RECEIVED_PARAMS)

    train_zero_shot(PARAMS, dataset, args.train_ratio, args.models, args.split_scheme, args.use_gpu)

