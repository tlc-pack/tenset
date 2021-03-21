"""Print all tasks"""

import pickle

from common import TO_MEASURE_PROGRAM_FOLDER

tasks = pickle.load(open(f"{TO_MEASURE_PROGRAM_FOLDER}/all_tasks.pkl", "rb"))

for i, t in enumerate(tasks):
    print("=" * 60)
    print(f"idx: {i}")
    print(f"flop_ct: {t.compute_dag.flop_ct}")
    print(f"workload_key: {t.workload_key}")
    print("")
    print(t.compute_dag)

