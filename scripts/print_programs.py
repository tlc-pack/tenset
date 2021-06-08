"""Print programs in a measurement record file.

Usage:
# Print all programs
python3 print_programs.py --filename 'dataset/measure_records/e5-2673/([12b88bedece6984af589a28b43e0f3c4,1,56,56,64,3,3,64,128,1,1,1,128,1,28,28,128],llvm).json'
# Print a specific program
python3 print_programs.py --filename 'dataset/measure_records/e5-2673/([12b88bedece6984af589a28b43e0f3c4,1,56,56,64,3,3,64,128,1,1,1,128,1,28,28,128],llvm).json' --idx 31
"""

import argparse

import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler import ComputeDAG, LayoutRewriteOption
from tvm.auto_scheduler.measure import recover_measure_input
from tvm.auto_scheduler.workload_registry import workload_key_to_tensors

from common import load_and_register_tasks


def print_program(index, inp, res):
    inp = recover_measure_input(inp, True)
    print("=" * 60)
    print(f"Index: {index}")
    print(f"Time cost (second): {res.costs}")
    print("Program:")
    print(inp.state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--idx", type=int)
    args = parser.parse_args()

    print("Load tasks...")
    tasks = load_and_register_tasks()

    inputs, results = auto_scheduler.RecordReader(args.filename).read_lines()
    if args.idx is None:
        for i in range(len(inputs)):
            print_program(i, inputs[i], results[i])
    else:
        print_program(args.idx, inputs[args.idx], results[args.idx])

