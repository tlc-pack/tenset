## Dataset organization

The dataset is stored under `/scripts/dataset` folder.

- dataset
  - network_info: The metadata for networks
     - {network_key}.model.pkl: The relay IR of a network. One network per file.
     - {network_key}.task.pkl: The tasks and their weights in a network. One network per file.
  - to_measure_programs: The generated random programs for measurement
     - all_tasks.pkl: A file containing all tasks
     - {task_key}.json: The measurement records (programs can be derived from measurement records). One file per task.
  - measure_records:
     - e5-2673: measurement records collected on an Intel e5-2673.
     - ...: 


## Data Collection Procedure

1. Dump metadata of all networks. The metadata includes all tasks and relay IR of a network
```
python3 dump_network_info.py
```
2. Dump all programs for measurement
```
python3 dump_programs.py
```

3. Measure all programs
```
python3 measure_programs.py
```

