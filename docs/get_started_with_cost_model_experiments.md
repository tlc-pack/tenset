# Get started 
This tutorial contains a minimal example of training a cost model and using it for search.

# Dataset
## Install and Download
1. Build and install this repo following the [install guide](https://tvm.apache.org/docs/install/index.html) of TVM.
2. Download dataset file.
    - You can download it from [google drive](https://drive.google.com/file/d/1hciRGyXcGY9fK_owgvlJow8P_l8xYIVJ/view?usp=sharing)
    - Or you can use the command line
    ```
    pip3 install gdown
    gdown https://drive.google.com/uc?id=1hciRGyXcGY9fK_owgvlJow8P_l8xYIVJ
    ```
3. Put `dataset_v3.3.zip` under `tvm-cost-model/scripts` and run `unzip dataset_v3.1.zip`
A new folder `dataset` will appear in `tvm-cost-model/scripts`.

## Dataset Content
see this [readme](https://github.com/merrymercy/tvm-cost-model/tree/main/scripts#dataset-organization)

# Example experiment

## Train a cost model and use it for search
Go to `tvm-cost-model/scripts`.

1. Make a dataset
You can either 
  - create a sampled smaller dataset for fast experiments.
  ```
  python3 make_dataset.py --logs dataset/measure_records/e5-2673/*.json --sample-in-files 100
  ```
- create a complete dataset by using all files. This takes a longer time and requires more memory.
```
python3 make_dataset.py --logs dataset/measure_records/e5-2673/*.json
```
2. Train a cost model
```
python3 train_model.py
```
3. Use the model for search
```
python3 tune_network.py --network resnet_50 --n-trials 100 --cost-model xgb-no-update --load-model xgb.pkl
```

