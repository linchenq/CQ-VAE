#!/bin/bash

# cuda: 1
python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 1 --task_name "aws_real0"

python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 2 --task_name "aws_real1"

python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 4 --task_name "aws_real2"

python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 8 --task_name "aws_real3"

python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 16 --task_name "aws_real4"

python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 32 --task_name "aws_real5"

python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 64 --task_name "aws_real6"