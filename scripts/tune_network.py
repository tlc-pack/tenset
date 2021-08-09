"""Tune a network"""
import argparse
import logging
import os
import random
import time

import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.contrib.graph_runtime as runtime
from tvm.auto_scheduler.utils import to_str_round

from dump_network_info import get_network_with_key
from common import str2bool, log_line, BenchmarkRecord

from search import random_search, local_search, default_search


def get_network(network_args):
    name, batch_size = network_args['network'], network_args['batch_size']
    if name in ['resnet_18', 'resnet_50', 'mobilenet_v2', 'mobilenet_v3',
                'wide_resnet_50', 'resnext_50', 'densenet_121', 'vgg_16']:
        network_key = (name, [(batch_size, 3, 224, 224)])
    elif name in ['inception_v3']:
        network_key = (name, [(batch_size, 3, 299, 299)])
    elif name in ['bert_tiny', 'bert_base', 'bert_medium', 'bert_large']:
        network_key = (name, [(batch_size, 128)])
    elif name == 'dcgan':
        network_key = (name, [(batch_size, 3, 64, 64)])
    else:
        raise ValueError("Invalid network: " + name)

    return get_network_with_key(network_key)


def get_tuning_option(tuning_args, target):
    n_trials, run_timeout, log_file = (
        tuning_args['n_trials'], tuning_args['run_timeout'], tuning_args['log_file'])

    if "cpu" in target.keys:
        measure_ctx = None
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=auto_scheduler.LocalRunner(
                repeat=10, number=1, enable_cpu_cache_flush=True, timeout=run_timeout),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
    elif "cuda" in target.keys:
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=run_timeout)
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
    else:
        raise NotImplementedError

    return tuning_opt, measure_ctx


def tune_and_evaluate(network_args, tuning_args, target, target_host, result_file, transfer_tune, search_type):
    mod, params, inputs = get_network(network_args)

    # Do auto-tuning
    if not tuning_args['eval_only']:
        # Delete existing log file to avoid reusing old measurement records
        if not tuning_args['continue_tuning'] and os.path.exists(log_file):
            print("Delete the existing log file %s" % log_file)
            os.system("rm -rf %s" % log_file)

        # Extract search tasks
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

        for idx, task in enumerate(tasks):
            print(
                "========== Task %d  (workload key: %s...) =========="
                % (idx, task.workload_key[:20])
            )
            print(task.compute_dag)

        tuning_opt, measure_ctx = get_tuning_option(tuning_args, target)

        # Run search
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights,
            load_model_file=tuning_args['load_model'], load_log_file=tuning_args['log_file'])
        policy = 'sketch.%s' % tuning_args['cost_model']

        if not transfer_tune:
            tuner.tune(tuning_opt, search_policy=policy)
        else:
            tuner.transfer_tune(tuning_opt, search_policy=policy)

    best_by_targetkey, _ = local_search(log_file)
    if search_type == "random":
        prof_res = random_search(best_by_targetkey, network_args, target)
    else:
        prof_res = default_search(best_by_targetkey, network_args, target)

    # Dump results
    log_line(BenchmarkRecord(str(target.kind), 'gpu' if 'gpu' in target.keys else 'cpu',
                            'network',
                            "%s.B%d" % (network_args['network'], network_args['batch_size']),
                            'ours', 'default',
                            {"costs": prof_res}, time.time()), result_file)

    if measure_ctx is not None:
        del measure_ctx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Search task related arguments
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--eval-only", action='store_true')
    parser.add_argument("--continue-tuning", action='store_true')
    parser.add_argument("--transfer-tune", action="store_true")

    # Search strategy related arguments
    parser.add_argument("--cost-model", type=str, choices=['xgb', 'lgbm', 'random', 'xgb-no-update', 'lgbm-no-update', 'mlp', 'mlp-no-update', 'tab', 'tab-no-update'],
                        default='xgb', help="The type of program cost model")
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--load-model", type=str, help="Load pre trained cost model file")

    # Log file related arguments
    parser.add_argument("--log-file", type=str, help="Write measurement records to this log file")
    parser.add_argument("--n-lines", type=int,
                        help="Only use the first n lines of the log file")
    parser.add_argument("--result-file", type=str,
                        help="Save end-to-end latency to this file",
                        default="results.tsv")

    # Measurement related and other arguments
    parser.add_argument("--num-measure-per-iter", type=int, default=64,
                        help="The number of programs to be measured at each iteration")
    parser.add_argument("--build-timeout", type=int, default=10)
    parser.add_argument("--run-timeout", type=int, default=25)
    parser.add_argument("--early-stopping", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--search-type", type=str, default='default', choices=['random', 'default'])
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    #logging.basicConfig()
    #logging.getLogger('auto_scheduler').setLevel(logging.DEBUG)

    target = tvm.target.Target(args.target)
    if target.model == "unknown":
        log_file = args.log_file or "%s-B%d-%s.json" % (args.network, args.batch_size,
                                                        target.kind)
    else:
        log_file = args.log_file or "%s-B%d-%s-%s.json" % (args.network, args.batch_size,
                                                           target.kind, target.model)
 
    network_args = {
        "network": args.network,
        "batch_size": args.batch_size,
    }

    tuning_args = {
        "eval_only": args.eval_only,
        "continue_tuning": args.continue_tuning,
        "n_trials": args.n_trials,
        "log_file": log_file,
        "run_timeout": args.run_timeout,
        "cost_model": args.cost_model,
        "load_model": args.load_model,
        "n_lines": args.n_lines,
    }

    tune_and_evaluate(network_args, tuning_args, target, args.target_host,
                      args.result_file, args.transfer_tune, args.search_type)

