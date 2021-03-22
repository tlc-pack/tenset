"""Dump programs for all tasks"""

import argparse
import pickle
import gc
import glob
import time
import os

from tqdm import tqdm

from tvm import auto_scheduler

from common import NETWORK_INFO_FOLDER, TO_MEASURE_PROGRAM_FOLDER


def get_tasks():
    all_task_keys = set()
    all_tasks = []
    duplication = 0

    filenames = glob.glob(f"{NETWORK_INFO_FOLDER}/*.task.pkl")
    filenames.sort()

    for filename in tqdm(filenames):
        tasks, task_weights = pickle.load(open(filename, "rb"))
        for t in tasks:
            task_key = (t.workload_key, str(t.target.kind))

            if task_key not in all_task_keys:
                all_task_keys.add(task_key)
                all_tasks.append(t)
            else:
                duplication += 1

    return all_tasks


def dump_program(task, size, max_retry_iter=10):
    folder = TO_MEASURE_PROGRAM_FOLDER
    task_key = (task.workload_key, str(task.target.kind))
    filename = f"{folder}/{task_key}.json"

    if os.path.exists(filename):
        return

    policy = auto_scheduler.SketchPolicy(task,
            params={'evolutionary_search_num_iters': 1,
                    'evolutionary_search_population': min(size, 2048)}, verbose=0)

    states = policy.sample_initial_population()

    # Generate unique states
    all_state_str_set = set()
    all_state_list = []

    retry_ct = 0
    niter = 0

    while len(all_state_list) < size and retry_ct < max_retry_iter:
        ct_before = len(all_state_list)

        states = policy.evolutionary_search(states, len(states))
        for s in states:
            str_s = str(s)
            if str_s not in all_state_str_set:
                all_state_str_set.add(str_s)
                all_state_list.append(s)

            if len(all_state_list) >= size:
                break

        ct_after = len(all_state_list)

        if ct_before == ct_after:
            states = policy.sample_initial_population()
            retry_ct += 1
        else:
            retry_ct = 0

        print(niter, len(all_state_list))
        niter += 1
    all_state_list = all_state_list[:size]

    # Make measure inputs and results
    measure_inputs = []
    measure_results = []
    for state in all_state_list:
        measure_inputs.append(auto_scheduler.MeasureInput(task, state))
        measure_results.append(auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time()))

    # Dump to file
    auto_scheduler.save_records(filename, measure_inputs, measure_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-idx", type=int)
    parser.add_argument("--end-idx", type=int)
    args = parser.parse_args()

    # Read all tasks
    tasks = get_tasks()
    tasks.sort(key=lambda x: (str(x.target.kind), x.compute_dag.flop_ct, x.workload_key))

    # Dump the whole task index
    folder = TO_MEASURE_PROGRAM_FOLDER
    os.makedirs(folder, exist_ok=True)
    pickle.dump(tasks, open(f"{TO_MEASURE_PROGRAM_FOLDER}/all_tasks.pkl", "wb"))

    start_idx = args.start_idx or 0
    end_idx = args.end_idx or len(tasks)

    # Dump programs for all tasks
    for task in tqdm(tasks[start_idx:end_idx]):
        dump_program(task, size=3000)
        gc.collect()

