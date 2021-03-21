
## Dataset organization


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

