# TenSet: A Large-scale Program Performance Datasetfor Learned Tensor Compilers

TenSet is a large-scale multi-platform tensor program performance dataset.
TenSet contains 52 million program performance records collected from 6 hardware platforms.

This repo is based on a fork of [TVM](https://github.com/apache/tvm).

# Dataset Statics


- Hardware Platforms
| Hardware Platform | Cloud Instance | Other Comments |
| ----------------- | -------------- | -------------- |
| Intel Platinum 8272CL @ 2.60GHz (16 cores)  | Azure D32s\_v4 & AVX-512 |
| Intel E5-2673 v4 @ 2.30GHz (8 cores) | Azure F16s & AVX-2 |
| AMD EPYC 7452 @ 2.35GHz (4 cores) | Azure D16as\_v4  & AVX-2 |
| ARM Graviton2 (16 cores) | AWS c6g.4xlarge |  Neon |
| NVIDIA Tesla K80 | AWS p2.xlarge   | Kepler Architecture |
| NVIDIA Tesla T4  | AWS g4dn.xlarge | Turing Architecture |


# Get Started with the Cost Model Experiments
See this [tutorial](docs/get_started_with_cost_model_experiments.md).

# Inspect Tasks and Programs in the Dataset
Follow the above tutorial to download the dataset.

- Print a task
```bash
cd scripts
python3 print_all_tasks.py --idx 1264
```

output:
```python
idx: 1264
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
python3 print_programs.py --idx 31
```

output:
```python
idx: 31
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

