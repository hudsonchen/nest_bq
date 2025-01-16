#!/bin/bash

# for seed in {0..40}
# do
#   for eps in 0.00003 0.00001
#   do
#     /home/zongchen/miniconda3/envs/cbq/bin/python evppi.py --seed $seed --multi_level --eps $eps
#   done
# done


# for seed in {0..10}
# do
#   for eps in 0.1 0.03 0.01 0.003 0.001
#   do
#     /home/zongchen/miniconda3/envs/cbq/bin/python evppi.py --seed $seed --eps $eps --kernel 'rbf'
#   done
# done


for seed in {0..10}
do
  for eps in 0.1 0.03 0.01 0.003
  do
    /home/zongchen/miniconda3/envs/cbq/bin/python evppi.py --seed $seed --eps $eps --kernel 'rbf' --qmc
  done
done