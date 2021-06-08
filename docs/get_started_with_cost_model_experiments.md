# Get started 
This tutorial contains a minimal example of training a cost model and using it for search.

## Install and Download the Dataset
### Requirement
Build and install this fork of TVM following the [install guide](https://tvm.apache.org/docs/install/index.html) of TVM.

### Download and unzip
You can choose use either the CPU part or the GPU part.

#### CPU part
1. Download
  - You can download it from google drive with the link [dataset_cpu_v3.3.zip](https://drive.google.com/file/d/1JQwGEe8jCpuhZPnUxO0Sb1CJJ06uevy6/view?usp=sharing)
  - Or you can use the command line
    ```
    pip3 install gdown
    gdown https://drive.google.com/uc?id=1JQwGEe8jCpuhZPnUxO0Sb1CJJ06uevy6
    ```
2. Unzip  
  Put `dataset_cpu_v3.3.zip` under `tenset/scripts` and run `unzip dataset_cpu_v3.3.zip`.
  A new folder `dataset_cpu` will appear in `tenset/scripts`. Make 'dataset' as a softlink to it
  by `ln -s dataset_cpu dataset`.

#### GPU part
1. Download
  - You can download it from google drive with the link [dataset_gpu_v3.2.zip](https://drive.google.com/file/d/1dlszmTBAXq9c_B7HcXRnBsqWOP76L-Jg/view?usp=sharing)
  - Or you can use the command line
    ```
    pip3 install gdown
    gdown gdown https://drive.google.com/uc?id=1dlszmTBAXq9c_B7HcXRnBsqWOP76L-Jg
    ```
2. Unzip  
  Put `dataset_gpu_v3.2.zip` under `tenset/scripts` and run `unzip dataset_gpu_v3.2.zip`.
  A new folder `dataset_gpu` will appear in `tenset/scripts`. Make 'dataset' as a softlink to it
  by `ln -s dataset_gpu dataset`.

## Example experiments

### Train a cost model and use it for search on CPU
Use the CPU part of the dataset and go to `tenset/scripts`.

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

### More experiments
More experiments will be added later.

