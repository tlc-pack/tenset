## Dataset organization

The dataset is stored under `tvm-cost-model/scripts/dataset` folder.

- dataset
  - `dataset/network_info`: The metadata for networks
     - `*.relay.pkl`: The relay IR of a network. One network per file.
         - For example, `('resnet_50', [(1, 3, 224, 224)]).relay.pkl` contains the relay IR of resnet_50 with input shape (1, 3, 224, 224).
     - `*.task.pkl`: The tasks and their weights in a network. One (network, targte) pair per file.
         - For example, `(('resnet_50', [(1, 3, 224, 224)]), 'llvm').task.pkl` contains the all tasks of resnet_50 on llvm backend.
  - `dataset/to_measure_programs`: The generated random programs for measurement
     - `all_tasks.pkl`: A file containing all tasks. It is used an an index for all tasks.
     - `*.json`: The randomly generated programs (schedules) for measurement. One file per task.
  - `dataset/measure_records`:
     - `e5-2666/*.json`: measurement records collected on an Intel e5-2666.
     - ...: 


This `dataset` folder contains model definitions, task information, and raw measurement records.
- `dataset/network_meta_data/*.model`  :  The model definition.
  - For example, `dataset/network_meta_data/resnet_50-B16.model` is the model definition of resnet-50 with batch size 16
- `dataset/network_meta_data/*.task` : The task information of a model.
  - For example, `dataset/network_meta_data/resnet_50-B16-llvm.model` contains the task information of resnet-50 with batch size 16 on CPU backend (LLVM).
- `dataset/*.json` : Raw measurement data.
  - For example, `dataset/resnet_50-B1-llvm-e5-2666.json` contains the raw measurement data of resnet-50 with batch size 1 on intel e5-2666 CPU.


## Data Collection Procedure

1. (about 30 mins) Dump metadata of all networks. The metadata includes all tasks and relay IR of a network.
```
python3 dump_network_info.py
```
2. (about 30 mins) Dump all programs for measurement
```
python3 dump_programs.py
```

3. Measure all programs
```
python3 measure_programs.py
```

