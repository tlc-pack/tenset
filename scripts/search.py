import argparse
import logging
import os
import random
import time
from collections import namedtuple
import heapq
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import tvm
from tvm.tir.expr import FloatImm
from tvm import relay, auto_scheduler
import tvm.contrib.graph_runtime as runtime
from tvm.auto_scheduler.measure_record import RecordToFile, load_records, save_records
from tvm.auto_scheduler.measure import MeasureInput, MeasureResult
from tvm.auto_scheduler.utils import decode_workload_key

from dump_network_info import get_network_with_key

LearningTask = namedtuple("LearningTask", ['workload_key', 'target'])
def input_to_learning_task(inp: MeasureInput):
    return LearningTask(inp.task.workload_key, str(inp.task.target))

def get_network(network_args):
    name, batch_size = network_args['network'], network_args['batch_size']
    if name in ['resnet_18', 'resnet_50', 'mobilenet_v2', 'mobilenet_v3',
                'wide_resnet_50', 'resnext_50', 'densenet_121']:
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


def get_workload_entry(best_records, target_key, workload_key):
    workload_hash, workload_args = decode_workload_key(workload_key)
    if target_key not in best_records:
        best_records[target_key] = {}
    if workload_hash not in best_records[target_key]:
        best_records[target_key][workload_hash] = {}
    return best_records[target_key][workload_hash], workload_hash, workload_args


def local_search(records, n_lines=None, n_lines_per_task=None):
    # global_search_space = {}
    if isinstance(records, pathlib.Path):
        records = str(records)

    if isinstance(records, str):
        records = load_records(records)

    if not records:
        return

    # Dict[str (target key),
    #   Dict[str (workload hash),
    #     Dict[tuple (workload args), pq (cost, State)]]]
    best_by_targetkey = {}
    best_by_model = {}

    counter = 0
    counter_per_task = {}
    for inp, res in records:
        task = input_to_learning_task(inp)
        if n_lines is not None and counter >= n_lines:
            break
        if task not in counter_per_task:
            counter_per_task[task] = 0
        if n_lines_per_task is not None and counter_per_task[task] >= n_lines_per_task:
            continue
        counter += 1
        counter_per_task[task] += 1
        if res.error_no != 0:
            continue

        costs = [x.value for x in res.costs if isinstance(x, FloatImm)]
        cost = np.mean(costs)

        # use target keys in tvm target system as key to build best map
        for k in inp.task.target.keys:
            entry, _, workload_args = get_workload_entry(
                best_by_targetkey, k, inp.task.workload_key
            )
            if workload_args not in entry:
                entry[workload_args] = []
            try:
                heapq.heappush(entry[workload_args], (cost, inp, res))
            except:
                print("same cost. continue")

        # use model as key to build best map
        entry, _, workload_args = get_workload_entry(
            best_by_model, inp.task.target.model, inp.task.workload_key
        )
        if workload_args not in entry:
            if inp.task.target.model != "unknown":
                entry[workload_args] = []
        if inp.task.target.model != "unknown":
            heapq.heappush(entry[workload_args], (cost, inp, res))

    return best_by_targetkey, best_by_model


def random_choose(global_search_space, top_k):
    inputs = []
    results = []
    for key in global_search_space:
        for hash in global_search_space[key]:
            for arg in global_search_space[key][hash]:
                candidates = heapq.nsmallest(top_k, global_search_space[key][hash][arg])
                selected = random.choice(candidates)
                inputs.append(selected[1])
                results.append(selected[2])
    return inputs, results


def measure(tmp_file, network_args, target, n_line_per_task=None):
    mod, params, inputs = get_network(network_args)
    with auto_scheduler.ApplyHistoryBest(tmp_file, n_line_per_task=n_line_per_task):
        with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(mod, target=target, params=params)
    ctx = tvm.context(str(target), 0)
    module = runtime.GraphModule(lib["default"](ctx))

    # Feed input data
    for name, shape, dtype in inputs:
        data_np = np.random.uniform(size=shape).astype(dtype)
        module.set_input(name, data_np)

    # Evaluate
    ftimer = module.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=3)
    prof_res = np.array(ftimer().results)
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res) * 1000, np.std(prof_res) * 1000))

    return np.mean(prof_res) * 1000


def default_search(global_search_space, network_args, target, tmp_file='tmp_log.json'):
    if os.path.exists(tmp_file):
        os.system("rm -rf %s" % tmp_file)
    inputs, results = random_choose(global_search_space, 1)
    save_records(tmp_file, inputs, results)
    all_best_cost = measure(tmp_file, network_args, target)
    return all_best_cost


def random_search(global_search_space, network_args, target, total_cts=30, top_k=3, tmp_file='tmp_log.json'):
    ct = 0
    start_search = time.time()
    if os.path.exists(tmp_file):
        os.system("rm -rf %s" % tmp_file)
    inputs, results = random_choose(global_search_space, 1)
    save_records(tmp_file, inputs, results)
    best_cost = measure(tmp_file, network_args, target)
    while ct < total_cts:
        if os.path.exists(tmp_file):
            os.system("rm -rf %s" % tmp_file)
        inputs, results = random_choose(global_search_space, top_k)
        save_records(tmp_file, inputs, results)
        cost = measure(tmp_file, network_args, target)
        if cost < best_cost:
            best_cost = cost
        ct += 1
        print(f"Cost for current round is {best_cost}. Time used is {time.time()-start_search}")
    return best_cost

