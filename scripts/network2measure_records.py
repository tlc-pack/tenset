"""Gather all measurement records of a network into a single file"""
import pickle

import tvm
from tvm.auto_scheduler.utils import run_cmd

from common import get_task_info_filename, get_measure_record_filename


if __name__ == "__main__":
    network_key = ("resnet_50", [(1, 3, 224,224)])
    target = "llvm -model=platinum-8272"

    # Read tasks of the network
    target = tvm.target.Target(target)
    task_info_filename = get_task_info_filename(network_key, target)
    tasks, task_weights = pickle.load(open(task_info_filename, "rb"))

    # Get measure record files
    measure_record_files = []
    for task in tasks:
        filename = get_measure_record_filename(task, target)
        measure_record_files.append(filename)

    # Concatenate them into a single file
    out_file = "tmp.json"
    run_cmd(f"rm -rf {out_file}")
    for filename in measure_record_files:
        run_cmd(f'cat "{filename}">> {out_file}')

