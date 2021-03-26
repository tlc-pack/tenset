"""Make a dataset file.

Usage:
python3 make_dataset.py dataset/measure_records/e5-2666/*.json
python3 make_dataset.py dataset/measure_records/e5-2666/*.json --sample-in-files 100
"""
import argparse
import glob
import random

from tvm import auto_scheduler

from common import load_and_register_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=str)
    parser.add_argument("--sample-in-files", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-file", type=str, default='dataset.pkl')
    parser.add_argument("--min-sample-size", type=int, default=48)
    args = parser.parse_args()

    load_and_register_tasks()

    files = args.logs
    if args.sample_in_files:
        random.seed(args.seed)
        files = random.sample(files, args.sample_in_files)

    auto_scheduler.dataset.make_dataset_from_log_file(
        files, args.out_file, args.min_sample_size)

