"""Print all tasks"""

import pickle

from common import load_and_register_tasks

tasks = load_and_register_tasks()

for i, t in enumerate(tasks):
    print("=" * 60)
    print(f"idx: {i}")
    print(f"flop_ct: {t.compute_dag.flop_ct}")
    print(f"workload_key: {t.workload_key}")
    print("")
    print(t.compute_dag)

