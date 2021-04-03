"""Dump programs for all tasks"""

import argparse
import pickle
import gc
import glob
import time
import os

from tqdm import tqdm

from tvm import auto_scheduler

from common import load_and_register_tasks, get_to_measure_filename


def dump_program(task, size, max_retry_iter=10):
    filename = get_to_measure_filename(task)
    if os.path.exists(filename):
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    policy = auto_scheduler.SketchPolicy(task,
            params={'evolutionary_search_num_iters': 1,
                    'evolutionary_search_population': min(size, 2560)}, verbose=0)

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
    parser.add_argument("--size", type=int, default=4000)
    args = parser.parse_args()

    tasks = load_and_register_tasks()

    start_idx = args.start_idx or 0
    end_idx = args.end_idx or len(tasks)

    # Dump programs for all tasks
    for task in tqdm(tasks[start_idx:end_idx]):
        dump_program(task, size=args.size)
        gc.collect()

