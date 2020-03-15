#!/bin/bash

# cuda: 0
python train.py --device "cuda:0" --sample_step 64 --real_sample 16 --tau 2 --min_tau 0.5 --task_name "tau0"

python train.py --device "cuda:0" --sample_step 64 --real_sample 16 --tau 5 --min_tau 0.5 --task_name "tau1"

python train.py --device "cuda:0" --sample_step 64 --real_sample 16 --tau 20 --min_tau 5 --task_name "tau2"

python train.py --device "cuda:0" --sample_step 64 --real_sample 16 --tau 100 --min_tau 5 --task_name "tau3"

python train.py --device "cuda:0" --sample_step 64 --real_sample 16 --tau 0.5 --min_tau 0.5 --task_name "tau4"

python train.py --device "cuda:0" --sample_step 64 --real_sample 16 --tau 2 --min_tau 2 --task_name "tau5"

python train.py --device "cuda:0" --sample_step 64 --real_sample 16 --tau 5 --min_tau 5 --task_name "tau6"

python train.py --device "cuda:0" --sample_step 64 --real_sample 16 --tau 50 --min_tau 50 --task_name "tau7"