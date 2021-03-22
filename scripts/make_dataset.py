"""Make a dataset file.

Usage:
python3 make_dataset.py 1060.json v100.json
python3 make_dataset.py bert-B1-llvm.json --load-network
"""
import argparse
import glob

from tvm import auto_scheduler

import common
from measure_programs import load_and_register_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", type=str, default='dataset.pkl')
    parser.add_argument("--min-sample-size", type=int, default=48)
    args = parser.parse_args()

    load_and_register_tasks()

    log_files = glob.glob("dataset/measure_records/e5-2673/*.json")

    auto_scheduler.dataset.make_dataset_from_log_file(
        log_files, args.out_file, args.min_sample_size)

