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


def preset_batch_size_1():
    network_keys = []

    # resnet_18 and resnet_50
    for batch_size in [1]:
        for image_size in [224, 240, 256]:
            for layer in [18, 50]:
                network_keys.append((f'resnet_{layer}',
                                    [(batch_size, 3, image_size, image_size)]))

    # mobilenet_v2
    for batch_size in [1]:
        for image_size in [224, 240, 256]:
            for name in ['mobilenet_v2', 'mobilenet_v3']:
                network_keys.append((f'{name}',
                                    [(batch_size, 3, image_size, image_size)]))

    # wide-resnet
    for batch_size in [1]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append((f'wide_resnet_{layer}',
                                    [(batch_size, 3, image_size, image_size)]))

    # resnext
    for batch_size in [1]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append((f'resnext_{layer}',
                                    [(batch_size, 3, image_size, image_size)]))

    # inception-v3
    for batch_size in [1]:
        for image_size in [299]:
            network_keys.append((f'inception_v3',
                                [(batch_size, 3, image_size, image_size)]))

    # densenet
    for batch_size in [1]:
        for image_size in [224, 240, 256]:
            network_keys.append((f'densenet_121',
                                [(batch_size, 3, image_size, image_size)]))

    # resnet3d
    for batch_size in [1]:
        for image_size in [112, 128, 144]:
            for layer in [18]:
                network_keys.append((f'resnet3d_{layer}',
                                    [(batch_size, 3, image_size, image_size, 16)]))

    # bert
    for batch_size in [1]:
        for seq_length in [64, 128, 256]:
            for scale in ['tiny', 'base', 'medium', 'large']:
                network_keys.append((f'bert_{scale}',
                                    [(batch_size, seq_length)]))

    # dcgan
    for batch_size in [1]:
        for image_size in [64, 80, 96]:
            network_keys.append((f'dcgan',
                                [(batch_size, 3, image_size, image_size)]))

    return network_keys

def preset_exclude(preset):
    network_keys = []

    # resnet_18 and resnet_50
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            if preset == 'no-resnet_50':
                print("no-resnet_50")
                for layer in [18]:
                    network_keys.append((f'resnet_{layer}',
                                         [(batch_size, 3, image_size, image_size)]))
            else:
                for layer in [18, 50]:
                    network_keys.append((f'resnet_{layer}',
                                         [(batch_size, 3, image_size, image_size)]))

    # mobilenet_v2
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            if preset == 'no-mobilenet_v2':
                for name in ['mobilenet_v3']:
                    network_keys.append((f'{name}',
                                         [(batch_size, 3, image_size, image_size)]))
            else:
                for name in ['mobilenet_v2', 'mobilenet_v3']:
                    network_keys.append((f'{name}',
                                         [(batch_size, 3, image_size, image_size)]))

    # wide-resnet
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append((f'wide_resnet_{layer}',
                                     [(batch_size, 3, image_size, image_size)]))

    # resnext
    if preset != 'no-resnext_50':
        for batch_size in [1, 4, 8]:
            for image_size in [224, 240, 256]:
                for layer in [50]:
                    network_keys.append((f'resnext_{layer}',
                                         [(batch_size, 3, image_size, image_size)]))

    # inception-v3
    for batch_size in [1, 2, 4]:
        for image_size in [299]:
            network_keys.append((f'inception_v3',
                                 [(batch_size, 3, image_size, image_size)]))

    # densenet
    for batch_size in [1, 2, 4]:
        for image_size in [224, 240, 256]:
            network_keys.append((f'densenet_121',
                                 [(batch_size, 3, image_size, image_size)]))

    # resnet3d
    for batch_size in [1, 2, 4]:
        for image_size in [112, 128, 144]:
            for layer in [18]:
                network_keys.append((f'resnet3d_{layer}',
                                     [(batch_size, 3, image_size, image_size, 16)]))

    # bert
    for batch_size in [1, 2, 4]:
        for seq_length in [64, 128, 256]:
            if preset == 'no-bert_tiny':
                for scale in ['base', 'medium', 'large']:
                    network_keys.append((f'bert_{scale}',
                                         [(batch_size, seq_length)]))
            elif preset == 'no-bert_base':
                for scale in ['tiny', 'medium', 'large']:
                    network_keys.append((f'bert_{scale}',
                                         [(batch_size, seq_length)]))
            else:
                for scale in ['tiny', 'base', 'medium', 'large']:
                    network_keys.append((f'bert_{scale}',
                                         [(batch_size, seq_length)]))

    # dcgan
    for batch_size in [1, 4, 8]:
        for image_size in [64, 80, 96]:
            network_keys.append((f'dcgan',
                                 [(batch_size, 3, image_size, image_size)]))

    return network_keys

def get_hold_out_task(target):
    network_keys = []

    # resnet_18 and resnet_50
    for layer in [18, 50]:
        network_keys.append((f'resnet_{layer}',[(1, 3, 224, 224)]))

    # mobilenet_v2
    network_keys.append(('mobilenet_v2', [(1, 3, 224, 224)]))

    # resnext
    network_keys.append(('resnext_50', [(1, 3, 224, 224)]))

    # bert
    for scale in ['tiny', 'base']:
        network_keys.append((f'bert_{scale}', [(1, 128)]))

    all_tasks = []
    exists = set()  # a set to remove redundant tasks
    print("hold out...")
    for network_key in tqdm(network_keys):
        # Read tasks of the network
        task_info_filename = get_task_info_filename(network_key, target)
        tasks, _ = pickle.load(open(task_info_filename, "rb"))
        for task in tasks:
            if task.workload_key not in exists:
                exists.add(task.workload_key)
                all_tasks.append(task)

    return all_tasks



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", type=str)
    parser.add_argument("--preset", type=str, choices=['batch-size-1', 'no-resnet_50', 'no-mobilenet_v2',
                                                       'no-resnext_50', 'no-bert_tiny', 'no-bert_base'])
    parser.add_argument("--target", type=str, default="llvm -model=platinum-8272")
    parser.add_argument("--sample-in-files", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-file", type=str, default='dataset.pkl')
    parser.add_argument("--min-sample-size", type=int, default=48)
    parser.add_argument("--hold-out", action='store_true')
    parser.add_argument("--n-task", type=int)
    parser.add_argument("--n-measurement", type=int)

    args = parser.parse_args()

    random.seed(args.seed)

    if args.hold_out:
        target = tvm.target.Target(args.target)
        to_be_excluded = get_hold_out_task(target)
        print(len(to_be_excluded))
        network_keys = preset_exclude()

        all_tasks = []
        exists = set()  # a set to remove redundant tasks
        print("Load tasks...")
        for network_key in tqdm(network_keys):
            # Read tasks of the network
            task_info_filename = get_task_info_filename(network_key, target)
            tasks, _ = pickle.load(open(task_info_filename, "rb"))
            for task in tasks:
                if task not in to_be_excluded and task.workload_key not in exists:
                    exists.add(task.workload_key)
                    all_tasks.append(task)

        # Convert tasks to filenames
        files = []
        for task in all_tasks:
            filename = get_measure_record_filename(task, target)
            files.append(filename)


    elif args.n_task:
        network_keys = preset_exclude()
        target = tvm.target.Target(args.target)
        all_tasks = []
        exists = set()  # a set to remove redundant tasks
        print("Load tasks...")
        task_cnt = 0
        for network_key in tqdm(network_keys):
            # Read tasks of the network
            task_info_filename = get_task_info_filename(network_key, target)
            tasks, _ = pickle.load(open(task_info_filename, "rb"))
            random.shuffle(tasks)
            for task in tasks:
                if task_cnt < args.n_task and task.workload_key not in exists:
                    exists.add(task.workload_key)
                    all_tasks.append(task)
                    task_cnt += 1

        # Convert tasks to filenames
        files = []
        for task in all_tasks:
            filename = get_measure_record_filename(task, target)
            files.append(filename)


    else:

        if args.preset is not None:
            # Only use tasks from networks with batch-size = 1
            # Load tasks from networks
            if args.preset == 'batch-size-1':
                network_keys = preset_batch_size_1()
            else:
                print("precluding")
                network_keys = preset_exclude(args.preset)

            target = tvm.target.Target(args.target)
            all_tasks = []
            exists = set()   # a set to remove redundant tasks
            print("Load tasks...")
            for network_key in tqdm(network_keys):
                # Read tasks of the network
                task_info_filename = get_task_info_filename(network_key, target)
                tasks, _ = pickle.load(open(task_info_filename, "rb"))
                for task in tasks:
                    if task.workload_key not in exists:
                        exists.add(task.workload_key)
                        all_tasks.append(task)

            # Convert tasks to filenames
            files = []
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
        files, args.out_file, args.min_sample_size, n_measurement=args.n_measurement)

