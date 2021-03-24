# Collect measurement data on AWS clusters

## Setup
1. Launch instances with all dependency built and installed.
2. Name these instances as "c44X_00", "c44X_01", ... "c44X_50" with ssh/config

## Do collection
Run ```python3 collect_master.py``` in a master node (which can be your laptop).
This command will launch measurement jobs on all workers.
When a worker finishes its measurement job, it will save the results into a zip file and shutdown itself.

## Gather results
Run ```python3 collect_master.py``` in a master node.
This command will copy the zip file from all workers to the master node.
