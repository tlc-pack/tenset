"""Print network information"""

import ast
import glob
import pickle


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


if __name__ == "__main__":
    keys = get_all_network_key()

    total_tasks = 0

    for i, key in enumerate(keys):
        print(f"========== Network {i+1}/{len(keys)}: {key} ==========")
        num_tasks = print_network_info(key)

        total_tasks += num_tasks

    print("#tasks", total_tasks)

