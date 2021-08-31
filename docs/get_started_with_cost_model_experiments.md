# Get started 
This tutorial contains a minimal example of training a cost model and using it for search.

## Install and Download the Dataset
### Requirement
Build and install this fork of TVM following the [install guide](https://tvm.apache.org/docs/install/index.html) of TVM.
If the latest TVM documentation does not work, you can follow the [old copy](install/from_source.rst) in this repo.

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
  - You can download it from google drive with the link [dataset_gpu_v3.3.zip](https://drive.google.com/file/d/1jqHbmvXUrLPDCIqJIaPee_atsPc0ZFFK/view?usp=sharing)
  - Or you can use the command line
    ```
    pip3 install gdown
    gdown https://drive.google.com/uc?id=1jqHbmvXUrLPDCIqJIaPee_atsPc0ZFFK
    ```
2. Unzip  
  Put `dataset_gpu_v3.3.zip` under `tenset/scripts` and run `unzip dataset_gpu_v3.3.zip`.
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
1. Dataset
    - Make a dataset with a certain number of tasks and a certain number of measurements per task. E.g.
        ```
        python3 make_dataset.py --logs dataset/measure_records/e5-2673/*.json --n-task 200 --n-measurement 200
        ```
    - Make a hold-out dataset. E.g. hold out all tasks appeared in resnet-50
        ```
        python3 make_dataset.py --logs dataset/measure_records/e5-2673/*.json --hold-out resnet-50
        ```
2. Model training
    - Specify the type of model. E.g.
      ```
      python3 train_model.py --models mlp
      ```
3. Search
    *Note that we might need to specify the target argument depending on the machine type.
    - No pre-trained model, and train a model online. E.g.
      ```
      python3 tune_network.py --network resnet_50 --n-trials 100 --cost-model xgb --target "llvm -mcpu=skylake-avx512"
      ```
    - Use a pre-trained model. E.g.
      ```
      python3 tune_network.py --network resnet_50 --n-trials 100 --cost-model mlp-no-update --load-model mlp.pkl
      ```
    - Use transfer-learning. E.g.
      ```
      python3 tune_network.py --network resnet_50 --n-trials 100 --cost-model mlp-no-update --load-model mlp.pkl --transfer-tune
      ```  
    


