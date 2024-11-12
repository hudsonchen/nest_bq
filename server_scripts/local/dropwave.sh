# For lookahead_EI_kq utility
for seed in {0..50}; do
    /home/zongchen/miniconda3/envs/look_ahead_bo/bin/python BO_new.py --utility lookahead_EI_kq --dataset dropwave --seed $seed --kernel matern --dim 2 --delta 0.01 --q 2 --iterations 60
done

# For lookahead_EI_mc utility
for seed in {0..50}; do
    /home/zongchen/miniconda3/envs/look_ahead_bo/bin/python BO_new.py --utility lookahead_EI_mc --dataset dropwave --seed $seed --kernel matern --dim 2 --delta 0.01 --q 2 --iterations 50
done

# For lookahead_EI_mlmc utility
for seed in {0..50}; do
    /home/zongchen/miniconda3/envs/look_ahead_bo/bin/python BO_new.py --utility lookahead_EI_mlmc --dataset dropwave --seed $seed --kernel matern --dim 2 --delta 0.2 --q 2 --iterations 50
done
