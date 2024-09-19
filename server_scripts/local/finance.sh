#!/bin/bash

for seed in {0..100}
do
  /home/zongchen/miniconda3/envs/cbq/bin/python finance.py --seed $seed
done
