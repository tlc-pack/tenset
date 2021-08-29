# Using pre-trained models on AWS c5.9xlarge  (or any other x86 CPUs with avx-512)
This is a quick tutorial on using a pre-trained models on AWS c5.9xlarge.

## Requirements
```
xgboost=1.2.1
torch==1.7.1
torchvision==0.8.2
```

## Steps
1. Build this fork of TVM following the [install guide](https://tvm.apache.org/docs/install/index.html) of TVM.
If the latest TVM documentation does not work, you can follow the [old copy](install/from_source.rst) in this repo.

2. Download the CPU dataset.
```
cd tenset/scripts
pip3 install gdown
gdown https://drive.google.com/uc?id=1JQwGEe8jCpuhZPnUxO0Sb1CJJ06uevy6
unzip dataset_cpu_v3.3.zip
ln -s dataset_cpu dataset
```

3. Sample a subset of the dataset and do featurization. The sampling is required because the c5.9xlarge instance does not have enough CPU memory to train a model on the full dataset.
```
python3 make_dataset.py --logs dataset/measure_records/platinum-8272/*.json --preset batch-size-1 --sample-in-files 500
```

Refernece output:
```
Load tasks...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:02<00:00, 18.08it/s]
Featurize measurement records...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:26<00:00, 18.59it/s]
No: 0   Task: LearningTask(workload_key='["bd2701f138c7fb53c6a136a3d8e9ee16", 1, 73, 73, 80, 6, 6, 192, 80, 1, 1, 1, 192, 1, 71, 71, 192]', target='llvm -keys=cpu -link-params=0 -mcpu=skylake-avx512 -model=platinum-8272')   Size:
4000
No: 1   Task: LearningTask(workload_key='["6b7583cf23c7c37d3212cad9d06e58c1", 1, 8, 8, 1280, 1, 1, 1280, 320, 1, 1, 1, 320, 1, 8, 8, 320]', target='llvm -keys=cpu -link-params=0 -mcpu=skylake-avx512 -model=platinum-8272')   Size:
4000
No: 2   Task: LearningTask(workload_key='["2350d19dc42a0665244368384c66b3a5", 1, 8, 8, 512, 3, 3, 512, 512, 1, 1, 1, 512, 1, 8, 8, 512]', target='llvm -keys=cpu -link-params=0 -mcpu=skylake-avx512 -model=platinum-8272')     Size:
4000
…
No: 498 Task: LearningTask(workload_key='["92d009cf6482ae0a815e6dcb60187533", 1, 30, 30, 120, 1, 1, 1, 120]', target='llvm -keys=cpu -link-params=0 -mcpu=skylake-avx512 -model=platinum-8272') Size: 21
Deleted
No: 499 Task: LearningTask(workload_key='["a12fbee7636d485f5404304276246b54", 1, 14, 14, 256, 1, 1, 256, 512, 1, 1, 1, 512, 1, 7, 7, 512]', target='llvm -keys=cpu -link-params=0 -mcpu=skylake-avx512 -model=platinum-8272')   Size:
4000
A dataset file is saved to dataset.pkl
```

4. Train a XGB model
```
python3 train_model.py --dataset dataset.pkl
```

Reference output:
```
Arguments: Namespace(dataset=['dataset.pkl'], models='xgb', seed=0, split_scheme='within_task', train_ratio=0.9, use_gpu=False)
Load all tasks...
Load dataset...
Train set: 1689022. Task 0 = LearningTask(workload_key='["bd2701f138c7fb53c6a136a3d8e9ee16", 1, 73, 73, 80, 6, 6, 192, 80, 1, 1, 1, 192, 1, 71, 71, 192]', target='llvm -keys=cpu -link-params=0 -mcpu=skylake-avx512 -model=platinum-8272')
Test set:  187679. Task 0 = LearningTask(workload_key='["bd2701f138c7fb53c6a136a3d8e9ee16", 1, 73, 73, 80, 6, 6, 192, 80, 1, 1, 1, 192, 1, 71, 71, 192]', target='llvm -keys=cpu -link-params=0 -mcpu=skylake-avx512 -model=platinum-8272')
Fit a xgb booster. Train size: 1689022
DEBUG:auto_scheduler:XGB iter:   0      te-rmse: 0.614730       tr-rmse: 0.618862       te-a-peak@1: 0.305379   tr-a-peak@1: 0.254582
DEBUG:auto_scheduler:XGB iter:  25      te-rmse: 0.103638       tr-rmse: 0.103122       te-a-peak@1: 0.755146   tr-a-peak@1: 0.718990
DEBUG:auto_scheduler:XGB iter:  50      te-rmse: 0.095500       tr-rmse: 0.094795       te-a-peak@1: 0.775069   tr-a-peak@1: 0.750568
DEBUG:auto_scheduler:XGB iter:  75      te-rmse: 0.091392       tr-rmse: 0.090687       te-a-peak@1: 0.790775   tr-a-peak@1: 0.763149
DEBUG:auto_scheduler:XGB iter: 100      te-rmse: 0.088617       tr-rmse: 0.087768       te-a-peak@1: 0.798640   tr-a-peak@1: 0.768572
DEBUG:auto_scheduler:XGB iter: 125      te-rmse: 0.086795       tr-rmse: 0.085874       te-a-peak@1: 0.808909   tr-a-peak@1: 0.773970
DEBUG:auto_scheduler:XGB iter: 150      te-rmse: 0.085503       tr-rmse: 0.084540       te-a-peak@1: 0.814928   tr-a-peak@1: 0.783759
DEBUG:auto_scheduler:XGB iter: 175      te-rmse: 0.084452       tr-rmse: 0.083455       te-a-peak@1: 0.815472   tr-a-peak@1: 0.794533
DEBUG:auto_scheduler:XGB iter: 200      te-rmse: 0.083567       tr-rmse: 0.082463       te-a-peak@1: 0.821549   tr-a-peak@1: 0.802378
DEBUG:auto_scheduler:XGB iter: 225      te-rmse: 0.082765       tr-rmse: 0.081631       te-a-peak@1: 0.821663   tr-a-peak@1: 0.800199
DEBUG:auto_scheduler:XGB iter: 250      te-rmse: 0.082108       tr-rmse: 0.080895       te-a-peak@1: 0.821506   tr-a-peak@1: 0.811268
DEBUG:auto_scheduler:XGB iter: 275      te-rmse: 0.081427       tr-rmse: 0.080149       te-a-peak@1: 0.826362   tr-a-peak@1: 0.813433
Save model to xgb.pkl
Test set sizes: [400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 6, 400, 400,
400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 77, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 8, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 11, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 7, 400, 400, 400, 11, 400, 400, 11, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 7, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
400, 400, 400, 400, 400, 400, 400, 73, 84, 400, 8, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 7, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 7, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 88, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 67, 400, 400, 7, 400, 400, 400, 400, 400, 400, 400, 400, 400]
xgb {'RMSE': '0.080881', 'R^2': '0.743378', 'pairwise comparision accuracy': '0.846439', 'mape': '29587344424.402508', 'average peak score@1': '0.829146', 'average peak score@5': '0.904636'}
------------------------------------------------------------
Model: xgb
RMSE: 0.0809
R^2: 0.7434
pairwise comparision accuracy: 0.8464
mape: 29587344424.4025
average peak score@1: 0.8291
average peak score@5: 0.9046
```

5. Search with the pretrained XGB model (100 trials)
```
python3 tune_network.py --network resnet_50 --n-trials 100 --cost-model xgb-no-update --load-model xgb.pkl --target "llvm -mcpu=skylake-avx512"
```

Reference output:
```
…
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.077 |          52.99 |      3 |
|    1 |        0.003 |          -0.00 |      3 |
|    2 |        0.136 |         755.97 |      3 |
|    3 |        0.208 |        1113.22 |      3 |
|    4 |        0.147 |         698.94 |      3 |
|    5 |        0.227 |        1020.46 |      3 |
|    6 |        0.166 |        1238.13 |      3 |
|    7 |        0.122 |         849.32 |      6 |
|    8 |        0.184 |         536.71 |      9 |
|    9 |        0.118 |         872.73 |      6 |
|   10 |        0.205 |        1129.03 |      3 |
|   11 |        0.162 |        1268.62 |      3 |
|   12 |        0.122 |         849.46 |      6 |
|   13 |        0.168 |         556.08 |      6 |
|   14 |        0.113 |         907.70 |      3 |
|   15 |        0.181 |        1276.71 |      3 |
|   16 |        0.162 |        1274.27 |      3 |
|   17 |        0.144 |         730.03 |      3 |
|   18 |        0.164 |        1408.33 |      6 |
|   19 |        0.113 |         911.20 |      3 |
|   20 |        0.043 |         602.03 |      3 |
|   21 |        0.026 |          70.83 |      3 |
|   22 |        0.193 |        1234.26 |      3 |
|   23 |        0.128 |         811.74 |      3 |
|   24 |        0.154 |        1334.94 |      3 |
|   25 |        0.194 |        1061.75 |      3 |
|   26 |        0.215 |         956.75 |      3 |
-------------------------------------------------
Estimated total latency: 7.977 ms       Trials: 102     Used time : 180 s       Next ID: -1
Mean inference time (std dev): 6.08 ms (0.10 ms)
```


6. Search without a pretrained model, or ansor-default (100 trials)
```
python3 tune_network.py --network resnet_50 --n-trials 100 --cost-model xgb --target "llvm -mcpu=skylake-avx512"
```

Reference output:
```
…
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.090 |          45.42 |      3 |
|    1 |        0.004 |          -0.00 |      3 |
|    2 |        0.326 |         316.08 |      3 |
|    3 |        0.318 |         726.35 |      9 |
|    4 |        0.148 |         693.23 |      3 |
|    5 |        0.486 |         475.70 |      3 |
|    6 |        0.303 |         678.70 |      3 |
|    7 |        0.187 |         551.82 |      6 |
|    8 |        0.347 |         283.90 |      6 |
|    9 |        0.129 |         799.96 |      6 |
|   10 |        0.461 |         502.19 |      3 |
|   11 |        0.786 |         262.04 |      3 |
|   12 |        0.164 |         632.94 |      6 |
|   13 |        0.352 |         264.98 |      3 |
|   14 |        0.313 |         328.68 |      3 |
|   15 |        1.399 |         165.41 |      3 |
|   16 |        0.256 |         804.42 |      3 |
|   17 |        0.219 |         480.11 |      6 |
|   18 |        0.237 |         978.54 |      3 |
|   19 |        0.608 |         169.71 |      3 |
|   20 |        0.247 |         105.59 |      3 |
|   21 |        0.027 |          67.51 |      3 |
|   22 |        0.591 |         402.09 |      3 |
|   23 |        0.189 |         546.81 |      3 |
|   24 |        0.365 |         564.01 |      3 |
|   25 |        0.759 |         271.09 |      3 |
|   26 |        0.985 |         208.67 |      3 |
-------------------------------------------------
Estimated total latency: 17.599 ms      Trials: 102     Used time : 194 s       Next ID: -1
Mean inference time (std dev): 17.71 ms (0.01 ms)
```

7. Search without a pretrained model, or ansor-default (5000 trials)
```
python3 tune_network.py --network resnet_50 --n-trials 5000 --cost-model xgb --target "llvm -mcpu=skylake-avx512"
```

Reference output: 
```
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.078 |          52.86 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.127 |         813.32 |    192 |
|    3 |        0.188 |        1232.77 |    320 |
|    4 |        0.121 |         848.58 |    128 |
|    5 |        0.239 |         966.94 |    128 |
|    6 |        0.177 |        1163.47 |    128 |
|    7 |        0.119 |         868.50 |    384 |
|    8 |        0.159 |         619.56 |    512 |
|    9 |        0.113 |         913.89 |    320 |
|   10 |        0.204 |        1135.63 |    128 |
|   11 |        0.150 |        1376.56 |    128 |
|   12 |        0.115 |         900.69 |    320 |
|   13 |        0.127 |         736.02 |    320 |
|   14 |        0.112 |         922.82 |    192 |
|   15 |        0.176 |        1311.57 |    128 |
|   16 |        0.160 |        1291.27 |    128 |
|   17 |        0.121 |         872.58 |    256 |
|   18 |        0.160 |        1443.41 |    320 |
|   19 |        0.112 |         921.06 |    128 |
|   20 |        0.064 |         404.97 |     64 |
|   21 |        0.022 |          82.32 |     64 |
|   22 |        0.175 |        1357.41 |    128 |
|   23 |        0.114 |         907.04 |    128 |
|   24 |        0.154 |        1336.39 |    128 |
|   25 |        0.162 |        1270.65 |    128 |
|   26 |        0.183 |        1120.69 |    128 |
-------------------------------------------------
Estimated total latency: 7.375 ms       Trials: 5015    Used time : 1918 s      Next ID: -1
Mean inference time (std dev): 5.91 ms (0.02 ms)
```

## Summary

| method            | ansor w/ pretrained cost model    | ansor w/o pretrained cost mdoel     | ansor w/o/ pretrained cost model |
| ----------------- | --- | --- | --- |
| n-trials          | 100 | 100 | 5000 |
| search time       | 180 s |194 s |  1918 s |
| inference latency | 6.08 ms | 17.71 ms | 5.91 ms|
