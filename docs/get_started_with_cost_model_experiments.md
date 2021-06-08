# Get started 
This tutorial contains a minimal example of training a cost model and using it for search.

## Install and Download the Dataset
1. Build and install this repo following the [install guide](https://tvm.apache.org/docs/install/index.html) of TVM.
2. Download dataset file.
    - You can download it from [google drive](https://drive.google.com/file/d/1JQwGEe8jCpuhZPnUxO0Sb1CJJ06uevy6/view?usp=sharing)
    - Or you can use the command line
    ```
    pip3 install gdown
    gdown https://drive.google.com/uc?id=1JQwGEe8jCpuhZPnUxO0Sb1CJJ06uevy6
    ```
3. Put `dataset_cpu_v3.3.zip` under `tenset/scripts` and run `unzip dataset_cpu_v3.3.zip`
A new folder `dataset` will appear in `tenset/scripts`.

## Example experiment

### Train a cost model and use it for search
Go to `tenset/scripts`.

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

