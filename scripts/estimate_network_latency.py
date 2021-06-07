"""Estimate the total latency of a network using measurement records"""
import argparse

import tvm
from tvm import  auto_scheduler

from tune_network import get_network


def estimaet_network(network_args, target, log_file):
    mod, params, inputs = get_network(network_args)

    # Extract search tasks
    target = tvm.target.Target(target)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    for idx, task in enumerate(tasks):
        print(
            "========== Task %d  (workload key: %s...) =========="
            % (idx, task.workload_key[:20])
        )
        print(task.compute_dag)

    # Read log files
    ctx = auto_scheduler.dispatcher.ApplyHistoryBest(log_file)
    target_key = target.keys[0]
    best_records = ctx.best_by_targetkey

    total_latency = 0
    for i in range(len(tasks)):
        entry, _, workload_args = \
            ctx.get_workload_entry(best_records, target_key, tasks[i].workload_key)
        total_latency += entry[workload_args][1] * task_weights[i]

    print(f"Estimated total latency: {total_latency * 1000:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    parser.add_argument("--log-file", type=str, required=True)
    args = parser.parse_args()

    network_args = {
        "network": args.network,
        "batch_size": args.batch_size,
    }

    estimaet_network(network_args, args.target, args.log_file)

