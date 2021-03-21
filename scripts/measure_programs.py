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


def remeasure_file(filename, target, target_host, batch_size):
    builder = auto_scheduler.measure.LocalBuilder()
    runner = auto_scheduler.measure.LocalRunner(
        timeout=15, repeat=8, number=1, enable_cpu_cache_flush=True)

    target = tvm.target.Target(target)
    folder = f"{MEASURE_RECORD_FOLDER}/{target.model}"
    os.makedirs(folder, exist_ok=True)
    log_filename = f"{folder}/{os.path.basename(filename)}"
    measurer = auto_scheduler.measure.ProgramMeasurer(
	builder,
	runner,
        [auto_scheduler.RecordToFile(log_filename)],
	verbose=1,
    )

    inputs, results = auto_scheduler.RecordReader(filename).read_lines()
    task = auto_scheduler.measure.recover_measure_input(inputs[0]).task
    task = auto_scheduler.SearchTask(
        workload_key=task.workload_key,
        target=target,
        target_host=target_host,
        hardware_params=task.hardware_params,
        layout_rewrite_option=task.layout_rewrite_option,
    )
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)

    for i in range(0, len(inputs), batch_size):
        inp_batch = inputs[i:min(len(inputs), i + batch_size)]
        res_batch = measurer.measure(task, empty_policy, inp_batch)


def load_workload_registry():
    tasks = pickle.load(open(f"{TO_MEASURE_PROGRAM_FOLDER}/task_registry.pkl", "rb"))

    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--target-host", type=str)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    # Load task registry
    print("Load workload registry...")
    load_workload_registry()

    # Remeasure all tasks
    filenames = glob.glob(f"{TO_MEASURE_PROGRAM_FOLDER}/*.json")
    filenames.sort()
    print(f"Load {len(filenames)} files")

    for i, filename in enumerate(filenames):
        with open("progress.txt", "a") as fout:
            fout.write(f"Begin {i}/{len(filenames)}: {time.time():.2f}\n")

        remeasure_file(filename, args.target, args.target_host, batch_size=args.batch_size)

        with open("progress.txt", "a") as fout:
            fout.write(f"End {i}/{len(filenames)}: {time.time():.2f}\n")

