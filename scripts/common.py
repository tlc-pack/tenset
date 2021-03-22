from collections import defaultdict, namedtuple
import pickle

import tvm
from tvm import relay, auto_scheduler
from tvm.auto_scheduler.utils import to_str_round


NETWORK_INFO_FOLDER = 'dataset/network_info'
TO_MEASURE_PROGRAM_FOLDER = 'dataset/to_measure_programs'
MEASURE_RECORD_FOLDER = 'dataset/measure_records'


def convert_to_nhwc(mod):
    """Convert to NHWC layout"""
    desired_layouts = {
        "nn.conv2d": ["NHWC", "default"],
        "nn.conv3d": ["NDHWC", "default"],
    }
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod


def load_and_register_tasks():
    tasks = pickle.load(open(f"{TO_MEASURE_PROGRAM_FOLDER}/all_tasks.pkl", "rb"))

    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors)

    return tasks


# The format for a line in resulst file
BenchmarkRecord = namedtuple("BenchmarkRecord",
                             ['device', 'backend', 'workload_type', 'workload_name',
                              'library', 'algorithm', 'value', 'time_stamp'])

def log_line(record, out_file):
    with open(out_file, 'a') as fout:
        fout.write("\t".join([to_str_round(x) for x in record]) + '\n')


def dtype2torch(x):
    import torch

    return {
        'float32': torch.float32
    }[x]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

