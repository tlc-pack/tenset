"""Measure all programs"""

import argparse
import glob
import os
import pickle
import time

from tqdm import tqdm

import tvm
from tvm import auto_scheduler

from common import TO_MEASURE_PROGRAM_FOLDER, MEASURE_RECORD_FOLDER

def make_measurer(run_timeout, repeat, number, enable_cpu_cache_flush,
                  verbose, log_filename):
    builder = auto_scheduler.measure.LocalBuilder()
    runner = auto_scheduler.measure.LocalRunner(
        timeout=run_timeout, repeat=repeat, number=number,
        enable_cpu_cache_flush=enable_cpu_cache_flush)
    measurer = auto_scheduler.measure.ProgramMeasurer(
	builder,
	runner,
        [auto_scheduler.RecordToFile(log_filename)],
	verbose=verbose,
    )
    return measurer


def remeasure_file(task_idx, reference_filename, target, target_host, batch_size, measurer_kwargs):
    # Make folder and log filename
    target = tvm.target.Target(target)
    folder = f"{MEASURE_RECORD_FOLDER}/{target.model}"
    os.makedirs(folder, exist_ok=True)
    log_filename = f"{folder}/{os.path.basename(reference_filename)}"

    # Make measuer
    measurer_kwargs['log_filename'] = log_filename
    measurer = make_measurer(**measurer_kwargs)

    # Read reference measurement inputs
    inputs, _ = auto_scheduler.RecordReader(reference_filename).read_lines()
    task = auto_scheduler.measure.recover_measure_input(inputs[0]).task
    task = auto_scheduler.SearchTask(
        workload_key=task.workload_key,
        target=target,
        target_host=target_host,
        hardware_params=task.hardware_params,
        layout_rewrite_option=task.layout_rewrite_option,
    )
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)

    # Do measurement
    for i in range(0, len(inputs), batch_size):
        print(f"===== task: {task_idx}\t programs: {i}/{len(inputs)} =====")
        inp_batch = inputs[i:min(len(inputs), i + batch_size)]
        res_batch = measurer.measure(task, empty_policy, inp_batch)

        timeout_ct = 0
        for res in res_batch:
            if res.error_no == auto_scheduler.measure.MeasureErrorNo.BUILD_TIMEOUT:
                timeout_ct += 1


def load_and_register_tasks():
    tasks = pickle.load(open(f"{TO_MEASURE_PROGRAM_FOLDER}/all_tasks.pkl", "rb"))

    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors)

    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--target-host", type=str)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-idx", type=int)
    parser.add_argument("--end-idx", type=int)
    args = parser.parse_args()

    # Load task registry
    print("Load all tasks...")
    tasks = load_and_register_tasks()

    start_idx = args.start_idx or 0
    end_idx = args.end_idx or len(tasks)

    # Remeasure all tasks
    for i in range(start_idx, end_idx):
        with open("progress.txt", "a") as fout:
            fout.write(f"Begin {i}/{len(tasks)}: {time.time():.2f}\n")
        task = tasks[i]

        # Set measurement arguments
        measurer_kwargs = {
            "run_timeout": 5,
            "repeat": 8,
            "number": 1,
            "enable_cpu_cache_flush": True,
            "verbose": 1,
        }
        if task.compute_dag.flop_ct >= 2416443392.0:
            measurer_kwargs['repeat'] = 3
        elif task.compute_dag.flop_ct >= 834928640.0:
            measurer_kwargs['repeat'] = 5
        elif task.compute_dag.flop_ct <= 2097152.0:
            measurer_kwargs['repeat'] = 10
        else:
            measurer_kwargs['repeat'] = 8

        # Run measurement
        task_key = (task.workload_key, str(task.target.kind))
        reference_filename = f"{TO_MEASURE_PROGRAM_FOLDER}/{task_key}.json"
        remeasure_file(i, reference_filename, args.target, args.target_host,
                       args.batch_size, measurer_kwargs)

        with open("progress.txt", "a") as fout:
            fout.write(f"End {i}/{len(tasks)}: {time.time():.2f}\n")

