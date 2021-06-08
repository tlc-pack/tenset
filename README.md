# TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers

TenSet is a large-scale multi-platform tensor program performance dataset.
TenSet contains 52 million program performance records collected from 6 hardware platforms.
This repo is based on a fork of [TVM](https://github.com/apache/tvm).

## Dataset Information

- Statics
  | Item | Number |
  | ---- | ------ |
  | Networks | 120 |
  | Hardware Platforms | 6 |
  | Tasks | 13,848 |
  | Measurement records | 51,577,248 |

- Hardware Platforms
  | Hardware Platform | Cloud Instance | Other Comments |
  | ----------------- | -------------- | -------------- |
  | Intel Platinum 8272CL @ 2.60GHz (16 cores)  | Azure D32s\_v4 | AVX-512 |
  | Intel E5-2673 v4 @ 2.30GHz (8 cores) | Azure F16s | AVX-2 |
  | AMD EPYC 7452 @ 2.35GHz (4 cores) | Azure D16as\_v4 | AVX-2 |
  | ARM Graviton2 (16 cores) | AWS c6g.4xlarge |  Neon |
  | NVIDIA Tesla K80 | AWS p2.xlarge   | Kepler Architecture |
  | NVIDIA Tesla T4  | AWS g4dn.xlarge | Turing Architecture |


## Get Started with the Cost Model Experiments
See this [tutorial](docs/get_started_with_cost_model_experiments.md).

## Organization
Follow the above tutorial to download the dataset. The dataset is stored under `tenset/scripts/dataset` folder.

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
   - `...`: other hardware platforms

## Inspect Tasks and Programs in the Dataset
Follow the above tutorial to download the dataset. You can then inspect the tasks and programs in the dataset

- Print a task
  ```bash
  cd scripts
  python3 print_all_tasks.py --idx 1264
  ```

  output:
  ```python
  Index: 1264
  flop_ct: 115806208.0
  workload_key: ["12b88bedece6984af589a28b43e0f3c4", 1, 56, 56, 64, 3, 3, 64, 128, 1, 1, 1, 128, 1, 28, 28, 128]
  Compute DAG:
  placeholder = PLACEHOLDER [1, 56, 56, 64]
  PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 57)) && (i2 >= 1)) && (i2 < 57)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
  placeholder = PLACEHOLDER [3, 3, 64, 128]
  Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
  placeholder = PLACEHOLDER [1, 1, 1, 128]
  T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
  T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)
  ```

- Print a program
  ```bash
  cd scripts
  python3 print_programs.py --filename 'dataset/measure_records/e5-2673/([12b88bedece6984af589a28b43e0f3c4,1,56,56,64,3,3,64,128,1,1,1,128,1,28,28,128],llvm).json' --idx 31
  ```

  output:
  ```python
  Index: 31
  Time cost (second): [0.000990787, 0.000826989, 0.00082599, 0.00083999, 0.000827089, 0.000831189, 0.00083599, 0.000853589]
  Program:
  Placeholder: placeholder, placeholder, placeholder
  parallel ax0.0@ax1.0@ax2.0@ (0,4)
    for i1 (0,57)
      for i2 ((floormod(ax0.outer.outer.ax1.outer.outer.fused.ax2.outer.outer.fused, 4)*14),15)
        for i3 (0,64)
          PaddedInput = ...
    for ax3.0 (0,2)
      for ax2.1 (0,7)
        for ax3.1 (0,8)
          Conv2dOutput auto_unroll: 16
          for rx.0 (0,3)
            for rc.0 (0,4)
              for ry.1 (0,3)
                for rc.1 (0,16)
                  for yy.3 (0,28)
                    vectorize ff.3 (0,8)
                      Conv2dOutput = ...
          for ax1.2 (0,28)
            vectorize ax3.2 (0,8)
              T_relu = ...
  ```


## License
The code is licensed under an [Apache-2.0](LICENSE) license.  
The dataset is licensed under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
