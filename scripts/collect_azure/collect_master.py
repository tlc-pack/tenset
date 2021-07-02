import os

from tqdm import tqdm


def run_cmd(cmd):
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        exit(ret)

def ssh_run(host, cmd):
    cmd = cmd.replace("\"", "\\\"")
    run_cmd(f"ssh -o StrictHostKeyChecking=no {host} \"{cmd}\"")


def ssh_tmux_run(host, cmd):
    cmd = f"tmux new-session -d \"{cmd}\""
    ssh_run(host, cmd)


n_tasks = 2308
n_machines = 10
tasks_per_machine = (n_tasks + n_machines - 1) // n_machines

if __name__ == "__main__":
    #target = "llvm -mcpu=skylake-avx512 -model=platinum-8272"
    target = "llvm -mcpu=core-avx2 -model=epyc-7452"
    #target = "llvm -mcpu=core-avx2 -model=e5-2673"

    print(f"Tasks_per_machine: {tasks_per_machine}")

    for i in tqdm(range(0, n_machines)):
        #host_name = f"azure-intel-avx512-{i:02d}"
        host_name = f"azure-amd-avx2-{i:02d}"
        #host_name = f"azure-intel-avx2-{i:02d}"

        ssh_run(host_name, "hostname")

        # fetch code
        ssh_run(host_name, "cd tvm-cost-model; git reset --hard 2df4c7a6b9a1; git pull;")

        ## run collection
        worker_commond = "source ~/.bashrc; cd tvm-cost-model/scripts; "\
                         "PYTHONPATH=~/tvm-cost-model/python python3 collect_azure/collect_worker.py "\
                        f"--start-idx {i} --end-idx {n_tasks} --step-idx {n_machines} "\
                        f"--target '{target}'"
        ssh_tmux_run(host_name, worker_commond)

