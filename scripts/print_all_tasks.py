"""Print all tasks.

Usage:
# Print all tasks
python3 print_all_tasks.py
# Print a specific task
python3 print_all_tasks.py --idx 10
"""
import argparse

from common import load_and_register_tasks

def print_task(index, task):
    print("=" * 60)
    print(f"Index: {index}")
    print(f"flop_ct: {task.compute_dag.flop_ct}")
    print(f"workload_key: {task.workload_key}")
    print("Compute DAG:")
    print(task.compute_dag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int)
    args = parser.parse_args()

    print("Load tasks...")
    tasks = load_and_register_tasks()

    if args.idx is None:
        for i, t in enumerate(tasks):
            print_task(i, t)
    else:
        print_task(args.idx, tasks[args.idx])

