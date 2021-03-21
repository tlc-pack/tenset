import tvm
from tvm import relay


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


def dtype2torch(x):
    import torch

    return {
        'float32': torch.float32
    }[x]

