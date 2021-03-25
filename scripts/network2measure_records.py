import pickle

import tvm
from tvm.auto_scheduler.utils import run_cmd

from common import NETWORK_INFO_FOLDER, MEASURE_RECORD_FOLDER


if __name__ == "__main__":
    network_key = ("resnet_50", [(1, 3, 224,224)])
    target = "llvm -model=e5-2666"
    out_file = "tmp.json"

    # Read tasks of the network
    target = tvm.target.Target(target)
    network_task_key = (network_key, str(target.kind))
    task_info_filename = f"{NETWORK_INFO_FOLDER}/{network_task_key}.task.pkl"
    tasks, task_weights = pickle.load(open(task_info_filename, "rb"))

    # Gather log files
    run_cmd(f"rm -rf {out_file}")
    for task in tasks:
        task_key = (task.workload_key, str(task.target.kind))
        filename = f"{MEASURE_RECORD_FOLDER}/{target.model}/{task_key}.json"
        filename = filename.replace('"', '\\"')
        run_cmd(f'cat "{filename}">> {out_file}')

