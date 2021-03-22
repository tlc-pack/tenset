"""Dump relay IR and task information for networks"""

import argparse
from collections import namedtuple
import gc
import multiprocessing
import os
import pickle

from tqdm import tqdm

import tvm
from tvm import relay
from tvm import auto_scheduler

from common import convert_to_nhwc, NETWORK_INFO_FOLDER, dtype2torch


def get_network_with_key(network_key):
    name, target, args = network_key

    if name in ['resnet_18', 'resnet_50', 'mobilenet_v2', 'mobilenet_v3',
                'wide_resnet_50', 'resnext_50', 'resnet3d_18']:
        import torch
        import torchvision.models as models   # torchvision>=0.9.0

        if name in ['resnet_18', 'resnet_50']:
            model = getattr(models, name.replace('_', ''))(pretrained=False)
        elif name == 'wide_resnet_50':
            model = getattr(models, 'wide_resnet50_2')(pretrained=False)
        elif name == 'resnext_50':
            model = getattr(models, 'resnext50_32x4d')(pretrained=False)
        elif name == 'mobilenet_v2':
            model = getattr(models, name)(pretrained=False)
        elif name == 'mobilenet_v3':
            model = getattr(models, name + "_large")(pretrained=False)
        elif name == 'resnet3d_18':
            model = models.video.r3d_18(pretrained=False)

        input_shape = args[0]
        dtype = 'float32'

        input_data = torch.randn(input_shape).type(dtype2torch(dtype))
        scripted_model = torch.jit.trace(model, input_data).eval()

        input_name = 'input0'
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        mod = convert_to_nhwc(mod)
        inputs = [(input_name, input_shape, dtype)]
    elif name in ['bert_tiny', 'bert_base', 'bert_medium', 'bert_large']:
        import torch
        import transformers  # pip3 install transformers==3.5 torch==1.7
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        config_dict = {
            "bert_tiny": transformers.BertConfig(num_hidden_layers=6, hidden_size=512,
                                                 intermediate_size=2048, num_attention_heads=8),
            "bert_base": transformers.BertConfig(num_hidden_layers=12, hidden_size=768,
                                                 intermediate_size=3072, num_attention_heads=12),
            "bert_medium": transformers.BertConfig(num_hidden_layers=12, hidden_size=1024,
                                                  intermediate_size=4096, num_attention_heads=16),
            "bert_large": transformers.BertConfig(num_hidden_layers=24, hidden_size=1024,
                                                  intermediate_size=4096, num_attention_heads=16),
        }

        configuration = config_dict[name]
        model = transformers.BertModel(configuration)

        input_shape = args[0]

        input_shape = input_shape
        input_name = 'input_ids'
        input_dtype = 'int64'
        A = torch.randint(10000, input_shape)

        model.eval()
        scripted_model = torch.jit.trace(model, [A], strict=False)

        input_name = 'input_ids'
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        mod = relay.transform.FastMath()(mod)
        mod = relay.transform.CombineParallelBatchMatmul()(mod)

        inputs = [(input_name, input_shape, input_dtype)]
    elif name == 'dcgan':
        import tvm.relay.testing

        output_shape = args[0]
        batch_size = output_shape[0]
        oshape = output_shape[1:]
        mod, params = relay.testing.dcgan.get_workload(
            batch_size=batch_size, oshape=oshape, layout="NHWC")
        inputs = [('data', (100,), 'float32')]
    else:
        raise ValueError("Invalid name: " + name)

    return mod, params, inputs


def dump_network(network_key):
    name, target, args = network_key

    folder = NETWORK_INFO_FOLDER
    os.makedirs(folder, exist_ok=True)
    model_filename = f"{folder}/{network_key}.model.pkl" 
    task_info_filename = f"{folder}/{network_key}.task.pkl"

    if os.path.exists(task_info_filename):
        return

    mod, params, inputs = get_network_with_key(key)

    # Dump network relay ir
    print(f"Dump relay ir for {network_key}...")
    mod_json = tvm.ir.save_json(mod)
    params_bytes = relay.save_param_dict(params)
    pickle.dump((mod_json, len(params_bytes), inputs), open(model_filename, "wb"))

    # Dump task information
    print(f"Dump task info for {network_key}...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, tvm.target.Target(target))
    pickle.dump((tasks, task_weights), open(task_info_filename, "wb"))


def build_network_keys():
    network_keys = []

    target = 'llvm'

    # resnet_18 and resnet_50
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            for layer in [18, 50]:
                network_keys.append((f'resnet_{layer}', target,
                                    [(batch_size, 3, image_size, image_size)]))

    # mobilenet_v2
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            for name in ['mobilenet_v2', 'mobilenet_v3']:
                network_keys.append((f'{name}', target,
                                    [(batch_size, 3, image_size, image_size)]))

    # wide-resnet
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append((f'wide_resnet_{layer}', target,
                                    [(batch_size, 3, image_size, image_size)]))

    # resnext
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append((f'resnext_{layer}', target,
                                    [(batch_size, 3, image_size, image_size)]))

    # resnet3d
    for batch_size in [1, 4, 8]:
        for image_size in [112, 128, 144]:
            for layer in [18]:
                network_keys.append((f'resnet3d_{layer}', target,
                                    [(batch_size, 3, image_size, image_size, 16)]))

    # bert
    for batch_size in [1, 2, 4]:
        for seq_length in [64, 128, 256]:
            for scale in ['tiny', 'base', 'medium', 'large']:
                network_keys.append((f'bert_{scale}', target,
                                    [(batch_size, seq_length)]))

    # dcgan
    for batch_size in [1, 4, 8]:
        for image_size in [64, 80, 96]:
            network_keys.append((f'dcgan', target, [(batch_size, 3, image_size, image_size)]))

    return network_keys


if __name__ == "__main__":
    network_keys = build_network_keys()

    for key in tqdm(network_keys):
        dump_network(key)
        gc.collect()

