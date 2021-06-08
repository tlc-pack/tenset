"""Print programs in a measurement record file.

Usage:
# Print all programs
python3 print_programs.py --filename dataset/to_measure_programs/'([fef3771fbad826271268252d597ca5fa,4,64,768,1,768,768,768,4,64,768,4,64,768],cuda).json'
# Print a specific program
python3 print_programs.py --filename dataset/to_measure_programs/'([fef3771fbad826271268252d597ca5fa,4,64,768,1,768,768,768,4,64,768,4,64,768],cuda).json'
"""

import argparse

import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler import ComputeDAG, LayoutRewriteOption
from tvm.auto_scheduler.measure import recover_measure_input
from tvm.auto_scheduler.workload_registry import workload_key_to_tensors

from common import load_and_register_tasks

def print_program(index, inp):
    inp = recover_measure_input(inp, True)
    print("=" * 60)
    print(f"idx: {index}")
    print(inp.state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--idx", type=int)
    args = parser.parse_args()

    print("Load tasks...")
    tasks = load_and_register_tasks()

    inputs, _ = auto_scheduler.RecordReader(args.filename).read_lines()
    if args.idx is None:
        for i, inp in enumerate(inputs):
            print_program(i, inp)
    else:
        print_program(args.idx, inputs[args.idx])

