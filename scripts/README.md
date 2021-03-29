## Dataset organization

The dataset is stored under `tvm-cost-model/scripts/dataset` folder.

- dataset
  - `dataset/network_info`: The metadata for networks
     - `*.relay.pkl`: The relay IR of a network. One network per file.
         - For example, `(resnet_50,[(1,3,224,224)]).relay.pkl` contains the relay IR of resnet_50 with input shape (1, 3, 224, 224).
     - `*.task.pkl`: The tasks and their weights in a network. One (network, targte) pair per file.
         - For example, `((resnet_50,[(1,3,224,224)]),llvm).task.pkl` contains all tasks of resnet_50 on llvm backend.
     - `all_tasks.pkl`: A file containing all tasks. It is used an an index for all tasks.
  - `dataset/to_measure_programs`: The generated random programs for measurement.
     - `*.json`: The randomly generated programs (schedules) for measurement. One file per task.
  - `dataset/measure_records`: Collected measurement records.
     - `e5-2666/*.json`: measurement records collected on an Intel e5-2673. One file per task.
     - `platinum-8272/*.json`: measurement records collected on an Intel platinum-8272. One file per task.
     - ...: 

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
python3 measure_programs.py --target "llvm -mcpu=core-avx2"
```

