#!/bin/bash

# cuda: 1

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 128 --task_name "real7"

# cuda: 2

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --real_sample 8 --sample_step 8 --task_name "realother0"

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --real_sample 8 --sample_step 16 --task_name "realother1"

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --real_sample 8 --sample_step 32 --task_name "realother2"

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --real_sample 8 --sample_step 64 --task_name "realother3"

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --real_sample 8 --sample_step 128 --task_name "realother4"

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --real_sample 8 --sample_step 256 --task_name "realother5"

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --real_sample 8 --sample_step 128 --task_name "realother6"

# cuda: 3
python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --sample_step 128 --real_sample 1 --task_name "ratio0"

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --sample_step 128 --real_sample 2 --task_name "ratio1"

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --sample_step 128 --real_sample 4 --task_name "ratio2"

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --sample_step 128 --real_sample 8 --task_name "ratio3"

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --sample_step 128 --real_sample 32 --task_name "ratio4"

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --sample_step 128 --real_sample 64 --task_name "ratio5"

python train.py --device "cuda:0" --tau 5 --min_tau 0.5 --sample_step 128 --real_sample 128 --task_name "ratio6"