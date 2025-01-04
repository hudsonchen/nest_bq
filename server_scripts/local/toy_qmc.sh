#!/bin/bash


for seed in {0..40}
do
  for eps in 0.1 0.03 0.01 0.003
  do
    /home/zongchen/miniconda3/envs/cbq/bin/python toy.py --kernel_x matern --kernel_theta matern --N_T_ratio 1.0 --d 1 --seed $seed --qmc
  done
done