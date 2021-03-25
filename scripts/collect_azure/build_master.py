"""Fetch and build code on workers"""

import multiprocessing
from collect_master import run_cmd, ssh_run, n_machines


def build_worker(host_name):
    ssh_run(host_name, "cd tvm-cost-model; git reset --hard 6d80e45af5; git pull; cd build; cmake ..; make -j30")

if __name__ == "__main__":

    host_name = [f"azure-intel-avx512-{i:02d}" for i in range(n_machines)]

    pool = multiprocessing.Pool(n_machines)
    pool.map(build_worker, host_name)

    print("Done")

