"""Print network information"""

import ast
import glob
import pickle

from tqdm import tqdm

from common import NETWORK_INFO_FOLDER

def print_network_info(network_key):
    folder = NETWORK_INFO_FOLDER

    task_info_filename = f"{folder}/{network_key}.task.pkl" 
    tasks, task_weights = pickle.load(open(task_info_filename, "rb"))

    for i, task in enumerate(tasks):
        print(f"----- Task {i+1}/{len(tasks)} -----")
        print(f"workload key: {task.workload_key}")
        print(task.compute_dag)

    return len(tasks)


def get_all_network_key():
    folder = NETWORK_INFO_FOLDER
    keys = []

    for filename in glob.glob(f"{folder}/*.model.pkl"):
        prefix = folder + "/"
        suffix = ".model.pkl"
        key = filename[len(prefix):-len(suffix)]
        keys.append(ast.literal_eval(key))

    return keys


def print_all_tasks():
    keys = get_all_network_key()

    total_tasks = 0

    for i, key in enumerate(keys):
        print(f"========== Network {i+1}/{len(keys)}: {key} ==========")
        num_tasks = print_network_info(key)

        total_tasks += num_tasks

    print("#tasks", total_tasks)


def count_all_tasks():
    all_task_keys = set()
    all_tasks = []
    duplication = 0

    for filename in tqdm(glob.glob(f"{NETWORK_INFO_FOLDER}/*.task.pkl")):
        tasks, task_weights = pickle.load(open(filename, "rb"))
        for t in tasks:
            task_key = (t.workload_key, str(t.target.kind))

            if task_key not in all_task_keys:
                all_task_keys.add(task_key)
                all_tasks.append(t)
            else:
                duplication += 1

    return len(all_tasks), duplication


if __name__ == "__main__":
    unique_ct, duplicated_ct = count_all_tasks()
    print(f"#Unique tasks: {unique_ct}")
    print(f"#Duplicated tasks: {duplicated_ct}")

