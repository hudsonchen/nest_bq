# Nested Kernel Quadrature

This repository contains the implementation of the code for the paper "Nested Expectations with Kernel Quadrature"

## Installation

To install the required packages, run the following command:
```
pip install -r requirements.txt
```

## Reproducing Results

### 1. Synthetic Experiment

To reproduce the results for the synthetic experiment (Figure 2 (Left)), run the following command:

`python toy.py --seed 0 --kernel_x matern --kernel_theta matern --N_T_ratio 1.0 --d 1`

You can vary the dimension by altering the argument '--d 1'.

### 2. Risk Management in Finance

To reproduce the results for the finance experiment (Figure 2 (Middle)), run:

`python finance.py --eps 0.003`

### 3. Health Economics.

To reproduce the results for the health economics experiment (Figure 2 (Right)), run:

`python evppi.py --eps 0.01 --kernel 'rbf'`

### 4. Bayesian Optimization
To reproduce the results for the bayesian optimization experiment (Figure 3), run:

`python BO.py --utility lookahead_EI_kq --dataset ackley --kernel matern --dim 2 --delta 0.01 --q 2 --iterations 30 --seed 0`