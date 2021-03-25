# Collect measurement data on Azure clusters

## Setup
1. Launch instances with all dependency built and installed.
2. Name these instances as "azure-intel-avx512-00", "azure-intel-avx512-01", ..., "azure-intel-avx512-19" with ssh/config

## Do collection
Run ```python3 collect_master.py``` in a master node (which can be your laptop).
This command will launch measurement jobs on all workers.
When a worker finishes its measurement job, it will save the results into a zip file.

## Gather results
Run ```python3 collect_master.py``` in a master node.
This command will copy the zip file from all workers to the master node.

