from tqdm import tqdm

from collect_master import run_cmd, ssh_run, n_machines


if __name__ == "__main__":
    for i in tqdm(range(n_machines)):
        print(i)
        host_name = f"c44X_{i:02d}"

        run_cmd(f"scp -o StrictHostKeyChecking=no {host_name}:tvm-cost-model/scripts/dataset_part_*.zip .")

