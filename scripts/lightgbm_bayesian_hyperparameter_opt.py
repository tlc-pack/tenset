"""Train a cost model with a dataset."""

import argparse
import logging
import pickle
import random

import torch
import numpy as np

import tvm
from tvm.auto_scheduler.utils import to_str_round
from tvm.auto_scheduler.cost_model import RandomModelInternal

from common import load_and_register_tasks, str2bool

from tvm.auto_scheduler.dataset import Dataset, LearningTask
from tvm.auto_scheduler.cost_model.lgbm_model import LGBModelInternal
from bayes_opt import BayesianOptimization
import pandas as pd
from tvm.auto_scheduler.cost_model.metric import (
    metric_rmse,
    metric_r_squared,
    metric_pairwise_comp_accuracy,
    metric_top_k_recall,
    metric_peak_score,
    metric_mape,
    random_mix,
)

import lightgbm as lgb

def evaluate_model(model, test_set):
    # make prediction
    prediction = model.predict(test_set)

    # compute weighted average of metrics over all tasks
    tasks = list(test_set.tasks())
    weights = [len(test_set.throughputs[t]) for t in tasks]
    #print("Test set sizes:", weights)

    rmse_list = []
    r_sqaured_list = []
    pair_acc_list = []
    mape_list = []
    peak_score1_list = []
    peak_score5_list = []


    for task in tasks:
        preds = prediction[task]
        labels = test_set.throughputs[task]

        rmse_list.append(np.square(metric_rmse(preds, labels)))
        r_sqaured_list.append(metric_r_squared(preds, labels))
        pair_acc_list.append(metric_pairwise_comp_accuracy(preds, labels))
        mape_list.append(metric_mape(preds, labels))
        peak_score1_list.append(metric_peak_score(preds, labels, 1))
        peak_score5_list.append(metric_peak_score(preds, labels, 5))

    rmse = np.sqrt(np.average(rmse_list, weights=weights))
    r_sqaured = np.average(r_sqaured_list, weights=weights)
    pair_acc = np.average(pair_acc_list, weights=weights)
    mape = np.average(mape_list, weights=weights)
    peak_score1 = np.average(peak_score1_list, weights=weights)
    peak_score5 = np.average(peak_score5_list, weights=weights)

    eval_res = {
        "RMSE": rmse,
        "R^2": r_sqaured,
        "pairwise comparision accuracy": pair_acc,
        "mape": mape,
        "average peak score@1": peak_score1,
        "average peak score@5": peak_score5,
    }
    return eval_res


def train_zero_shot(dataset, train_ratio, split_scheme):
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

    def lgb_eval(learning_rate,num_leaves, feature_fraction, bagging_fraction, bagging_freq, min_data_in_leaf, min_sum_hessian_in_leaf):
        params = {'boosting_type': 'gbdt'}
        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['bagging_freq'] = int(round(bagging_freq))
        params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
        
        model = LGBModelInternal(use_gpu=False, params=params)
        model.fit_base(train_set, valid_set=test_set)
        
        # Evaluate the model
        eval_res = evaluate_model(model, test_set)

        return -1 * eval_res['RMSE']
     
    lgbBO = BayesianOptimization(lgb_eval, {'learning_rate': (0.02, 0.2),
                                            'num_leaves': (24, 80),
                                            'feature_fraction': (0.6, 1),
                                            'bagging_fraction': (0.7, 1),
                                            'bagging_freq': (3, 10),
                                            'min_data_in_leaf': (0, 40),
                                            'min_sum_hessian_in_leaf':(0, 20),
                                            }, random_state=300)

    
    lgbBO.probe(
        params={
                'learning_rate': 0.05,
                'num_leaves': 31,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 0,
                'min_sum_hessian_in_leaf': 0,
            },
        lazy=True,
    )
    #n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
    #init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
    
    lgbBO.maximize(init_points=15, n_iter=15)
    
    model_auc=[]
    for model in range(len(lgbBO.res)):
        model_auc.append(lgbBO.res[model]['target'])
    
    # return best parameters
    best_result, opt_params = lgbBO.res[pd.Series(model_auc).idxmax()]['target'], lgbBO.res[pd.Series(model_auc).idxmax()]['params']

    print("best result: ", best_result, opt_params)

    def proc_params(params):
        params['boosting_type'] = 'gbdt'
        params['learning_rate'] = max(min(params['learning_rate'], 1), 0)
        params["num_leaves"] = int(round(params["num_leaves"]))
        params['feature_fraction'] = max(min(params['feature_fraction'], 1), 0)
        params['bagging_fraction'] = max(min(params['bagging_fraction'], 1), 0)
        params['bagging_freq'] = int(round(params['bagging_freq']))
        params['min_data_in_leaf'] = int(round(params['min_data_in_leaf']))
        params['min_sum_hessian_in_leaf'] = params['min_sum_hessian_in_leaf']
        return params 
    
    opt_params = proc_params(opt_params)

    model = LGBModelInternal(use_gpu=False, params=opt_params)
    # Train the model
    filename = "lightgbm_tuned.pkl"
    model.fit_base(train_set, valid_set=test_set)
    print("Save model to %s" % filename)
    model.save(filename)

    # Evaluate the model
    eval_res = evaluate_model(model, test_set)
    print(to_str_round(eval_res))

    # Print evaluation results
    print("-" * 60)
    print("Model: lightgbm_tuned")
    for key, val in eval_res.items():
        print("%s: %.4f" % (key, val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", type=str, default=["dataset.pkl"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-scheme",
        type=str,
        choices=["by_task", "within_task", "by_target"],
        default="within_task",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
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
    dataset = pickle.load(open(args.dataset[0], "rb"))
    for i in range(1, len(args.dataset)):
        tmp_dataset = pickle.load(open(args.dataset[i], "rb"))
        dataset.update_from_dataset(tmp_dataset)

    train_zero_shot(dataset, args.train_ratio, args.split_scheme)

