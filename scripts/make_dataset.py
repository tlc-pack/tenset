"""Make a dataset file.

Usage:
python3 make_dataset.py --logs dataset/measure_records/e5-2673/*.json
python3 make_dataset.py --logs dataset/measure_records/e5-2673/*.json --sample-in-files 100
python3 make_dataset.py --preset batch-size-1
"""
import argparse
import glob
import pickle
import random

from tqdm import tqdm
import tvm
from tvm import auto_scheduler

from common import (load_and_register_tasks, get_task_info_filename,
    get_measure_record_filename)

from dump_network_info import build_network_keys


def get_hold_out_task(target, network=None):
    network_keys = []

    if network == "resnet-50":
        print("precluding all tasks in resnet-50")
        for batch_size in [1, 4, 8]:
            for image_size in [224, 240, 256]:
                for layer in [50]:
                    network_keys.append((f'resnet_{layer}',
                                         [(batch_size, 3, image_size, image_size)]))
    else:
        # resnet_18 and resnet_50
        for layer in [18, 50]:
            network_keys.append((f'resnet_{layer}', [(1, 3, 224, 224)]))

        # mobilenet_v2
        network_keys.append(('mobilenet_v2', [(1, 3, 224, 224)]))

        # resnext
        network_keys.append(('resnext_50', [(1, 3, 224, 224)]))

        # bert
        for scale in ['tiny', 'base']:
            network_keys.append((f'bert_{scale}', [(1, 128)]))

    exists = set()
    print("hold out...")
    for network_key in tqdm(network_keys):
        # Read tasks of the network
        task_info_filename = get_task_info_filename(network_key, target)
        tasks, _ = pickle.load(open(task_info_filename, "rb"))
        for task in tasks:
            if task.workload_key not in exists:
                exists.add(task.workload_key)

    return exists



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", type=str)
    parser.add_argument("--target", nargs="+", type=str, default=["llvm -model=platinum-8272"])
    parser.add_argument("--sample-in-files", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-file", type=str, default='dataset.pkl')
    parser.add_argument("--min-sample-size", type=int, default=48)
    parser.add_argument("--hold-out", type=str, choices=['resnet-50', 'all_five'])
    parser.add_argument("--n-task", type=int)
    parser.add_argument("--n-measurement", type=int)
    parser.add_argument("--access_matrix", type=bool, default=True)


    args = parser.parse_args()

    random.seed(args.seed)

    files = []
    if args.hold_out or args.n_task:
        task_cnt = 0
        for target in args.target:
            target = tvm.target.Target(target)
            to_be_excluded = get_hold_out_task(target, args.hold_out)
            network_keys = build_network_keys()

            print("Load tasks...")
            print(f"target: {target}")
            all_tasks = []
            exists = set()  # a set to remove redundant tasks
            for network_key in tqdm(network_keys):
                # Read tasks of the network
                task_info_filename = get_task_info_filename(network_key, target)
                tasks, _ = pickle.load(open(task_info_filename, "rb"))
                for task in tasks:
                    if task.workload_key not in to_be_excluded and task.workload_key not in exists:
                        if not args.n_task or task_cnt < args.n_task:
                            exists.add(task.workload_key)
                            all_tasks.append(task)
                            task_cnt += 1

            # Convert tasks to filenames
            for task in all_tasks:
                filename = get_measure_record_filename(task, target)
                files.append(filename)

    else:
        # use all tasks
        print("Load tasks...")
        load_and_register_tasks()
        files = args.logs

    if args.sample_in_files:
        files = random.sample(files, args.sample_in_files)

    print("Featurize measurement records...")
    auto_scheduler.dataset.make_dataset_from_log_file(
        files, args.out_file, args.min_sample_size, access_matrix = args.access_matrix)

