#!/bin/bash

jobs_in_parallel=$(wc -l < "$1")
echo $jobs_in_parallel
echo $1

qsub -t 1-${jobs_in_parallel} /home/ucabzc9/Scratch/nest_bq/server_scripts/myriad/finance_base.sh "$1"

