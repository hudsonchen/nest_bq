#!/bin/bash

jobs_in_parallel=$(wc -l < "$1")
echo $jobs_in_parallel
echo $1

qsub -t 1-${jobs_in_parallel} /home/zongchen/nest_bq/server_scripts/cs_cluster/toy_base.sh "$1"

