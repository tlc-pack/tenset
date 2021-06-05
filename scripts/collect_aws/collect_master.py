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
n_machines = 50
tasks_per_machine = (n_tasks + n_machines - 1) // n_machines

if __name__ == "__main__":
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod -model=graviton2"

    print(f"Tasks_per_machine: {tasks_per_machine}")

    for i in tqdm(range(n_machines)):
        host_name = f"c64_{i:02d}"

        # fetch code
        ssh_run(host_name, "cd tenset; git reset --hard 15691e2d; git pull;")

        # run collection
        worker_commond = "source ~/.bashrc; cd tenset/scripts; "\
                         "PYTHONPATH=~/tenset/python python3 collect_aws/collect_worker.py "\
                        f"--start-idx {i} --end-idx {n_tasks} --step-idx {n_machines} "\
                        f"--target '{target}'"
        ssh_tmux_run(host_name, worker_commond)

