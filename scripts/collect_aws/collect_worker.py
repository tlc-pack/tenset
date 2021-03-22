import argparse

from collect_master import run_cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-idx", type=int)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--target", type=str)
    args = parser.parse_args()

    # clear old logs
    run_cmd("rm -rf dataset/measure_records")

    # warmup
    run_cmd(f'python3 measure_programs.py --start-idx 1 --end-idx 2 --target "{args.target}"')

    # clear warmup logs
    run_cmd("rm -rf dataset/measure_records")
    run_cmd("rm -rf progress.txt")

    # run_collect
    run_cmd(f"python3 measure_programs.py --start-idx {args.start_idx} " + 
            f'--end-idx {args.end_idx} --target "{args.target}"')

