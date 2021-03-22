from tqdm import tqdm

from collect_master import run_cmd, n_machines


if __name__ == "__main__":
    for i in tqdm(range(n_machines)):
        host_name = f"c44X_{i}"

        run_cmd(f"scp -o StrictHostKeyChecking=no {host_name}:tvm-cost-model/scripts/dataset_part_*.zip .")

