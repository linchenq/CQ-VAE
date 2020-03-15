#!/bin/bash

# cuda: 3
python train.py --device "cuda:3" --tau 5 --min_tau 0.5 --sample_step 64 --real_sample 1 --task_name "aws_ratio0"

python train.py --device "cuda:3" --tau 5 --min_tau 0.5 --sample_step 64 --real_sample 2 --task_name "aws_ratio1"

python train.py --device "cuda:3" --tau 5 --min_tau 0.5 --sample_step 64 --real_sample 4 --task_name "aws_ratio2"

python train.py --device "cuda:3" --tau 5 --min_tau 0.5 --sample_step 64 --real_sample 8 --task_name "aws_ratio3"

python train.py --device "cuda:3" --tau 5 --min_tau 0.5 --sample_step 64 --real_sample 16 --task_name "aws_ratio4"

python train.py --device "cuda:3" --tau 5 --min_tau 0.5 --sample_step 64 --real_sample 32 --task_name "aws_ratio5"

python train.py --device "cuda:3" --tau 5 --min_tau 0.5 --sample_step 64 --real_sample 64 --task_name "aws_ratio6"