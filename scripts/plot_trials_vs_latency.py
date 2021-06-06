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

def beam_choose(global_search_space, selected_candidates):
    inputs = []
    results = []
    for key in global_search_space:
        for hash in global_search_space[key]:
            for arg in global_search_space[key][hash]:
                if key in selected_candidates and hash in selected_candidates[key] and arg in selected_candidates[key][hash]:
                    candidate = selected_candidates[key][hash][arg]
                    inputs.append(candidate[1])
                    results.append(candidate[2])
                else:
                    best = heapq.nsmallest(1, global_search_space[key][hash][arg])[0]
                    inputs.append(best[1])
                    results.append(best[2])
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


def random_search(global_search_space, network_args, target, total_cts=30, top_k=3, tmp_file='tmp_log.json'):
    ct = 0
    # best_candidate = None
    best_cost = 1e20
    if os.path.exists(tmp_file):
        os.system("rm -rf %s" % tmp_file)
    start_all_best = time.time()
    inputs, results = random_choose(global_search_space, 1)
    save_records(tmp_file, inputs, results)
    all_best_cost = measure(tmp_file, network_args, target)
    best_cost = all_best_cost
    print(f"Time used for all best eval is {time.time()-start_all_best}")
    while ct < total_cts:
        start_search = time.time()
        if os.path.exists(tmp_file):
            os.system("rm -rf %s" % tmp_file)
        inputs, results = random_choose(global_search_space, top_k)
        save_records(tmp_file, inputs, results)
        cost = measure(tmp_file, network_args, target)
        if cost < best_cost:
            best_cost = cost
            # best_candidate = candidate
        ct += 1
        print(f"Cost for current round is {best_cost}. Time used is {time.time()-start_search}")
    # print(f"The best cost is {best_cost}")
    return best_cost, all_best_cost


def beam_search(global_search_space, network_args, target, beam=2, tmp_file='tmp_log.json'):
    best_cost = None
    selected_candidates = {}
    ct = 0
    for key in global_search_space:
        for hash in global_search_space[key]:
            for arg in global_search_space[key][hash]:
                print(f"starting task {ct}")
                if key not in selected_candidates:
                    selected_candidates[key] = {}
                if hash not in selected_candidates[key]:
                    selected_candidates[key][hash] = {}
                best_candidate = None
                candidates = heapq.nsmallest(beam, global_search_space[key][hash][arg])
                for candidate in candidates:
                    if os.path.exists(tmp_file):
                        os.system("rm -rf %s" % tmp_file)
                    selected_candidates[key][hash][arg] = candidate
                    inputs, results = beam_choose(global_search_space, selected_candidates)
                    save_records(tmp_file, inputs, results)
                    cost = measure(tmp_file, network_args, target)
                    if not best_candidate:
                        best_cost = cost
                        best_candidate = candidate
                    else:
                        if cost < best_cost:
                            best_cost = cost
                            best_candidate = candidate

                selected_candidates[key][hash][arg] = best_candidate
                ct += 1

    inputs, results = beam_choose(global_search_space, selected_candidates)
    save_records(tmp_file, inputs, results)
    cost = measure(tmp_file, network_args, target)
    print(f"The final cost is {cost}")
    return cost

def make_random_plot(network_args, log_file, target):
    mean_inf_time = []
    all_best = []
    for i in range(1, 100):
        print(f"Each task is measured {i} times")
        best_by_targetkey, _ = local_search(log_file, n_lines_per_task=i)
        cost, all_best_cost = random_search(best_by_targetkey, network_args, target)
        mean_inf_time.append(cost)
        all_best.append(all_best_cost)

    print(mean_inf_time)
    print(all_best)
    plt.plot(list(range(1, 100)), mean_inf_time)
    plt.savefig(f"{network_args['network']}_trials_vs_latency_random.png")
    
def make_beam_plot(network_args, log_file, target):
    mean_inf_time = []
    for i in range(1, 100):
        print(f"Each task is measured {i} times")
        best_by_targetkey, _ = local_search(log_file, n_lines_per_task=i)
        cost = beam_search(best_by_targetkey, network_args, target)
        mean_inf_time.append(cost)

    plt.plot(list(range(1, 100)), mean_inf_time)
    plt.savefig(f"{network_args['network']}_trials_vs_latency_beam.png")
    print(mean_inf_time)

def make_all_best_plot(network_args, log_file, target):
    mean_inf_time = []
    timestamp = []
    if isinstance(log_file, pathlib.Path):
        records = str(log_file)
    if isinstance(records, str):
        records = load_records(records)
    for i in range(0, 100, 2):
        print(f"Each task is measured {i} times")
        total_time = 0
        prev_task_end_time = 0
        total_search_time = 0
        counter_per_task = {}
        start_time_per_task = {}
        for inp, res in records:
            task = input_to_learning_task(inp)
            if task not in counter_per_task:
                total_search_time += res.timestamp - prev_task_end_time
                counter_per_task[task] = 0
                start_time_per_task[task] = res.timestamp
            if counter_per_task[task] == i:
                cur_task_time = res.timestamp - start_time_per_task[task]
                total_time += cur_task_time
                counter_per_task[task] += 1
                prev_task_end_time = res.timestamp
                continue
            elif counter_per_task[task] > i:
                continue
            else:
                counter_per_task[task] += 1

        total_time += total_search_time
        print(f"total search time: {total_search_time}")
        timestamp.append(total_time)
        # cost = measure(log_file, network_args, target, n_line_per_task=i)
        # mean_inf_time.append(cost)

    print(mean_inf_time)
    print(timestamp)
    # plt.plot(list(range(1, 100)), mean_inf_time[1:])
    # plt.savefig(f"{network_args['network']}_trials_vs_latency_all_best.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    parser.add_argument("--log-file", type=str)
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--search-type", type=str, default='random', choices=['random', 'all_best', 'beam'])
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    
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

    if args.search_type == 'all_best':
        make_all_best_plot(network_args, log_file, target)
    elif args.search_type == 'random':
        make_random_plot(network_args, log_file, target)
    elif args.search_type == 'beam':
        make_beam_plot(network_args, log_file, target)


