#!/bin/bash

for seed in {20..40}
do
  for eps in 0.03 0.01 0.003 0.001 0.0003 0.0001
  do
    /home/zongchen/miniconda3/envs/cbq/bin/python toy_mlmc.py --seed $seed --multi_level --eps $eps
  done
done

for seed in {0..40}
do
  for eps in 0.1 0.03 0.01 0.003
  do
    /home/zongchen/miniconda3/envs/cbq/bin/python toy_mlmc.py --seed $seed --eps $eps
  done
done