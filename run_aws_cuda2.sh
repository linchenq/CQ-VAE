#!/bin/bash

# cuda: 2

python train.py --device "cuda:2" --tau 5 --min_tau 0.5 --real_sample 4 --sample_step 4 --task_name "aws_realother0"

python train.py --device "cuda:2" --tau 5 --min_tau 0.5 --real_sample 4 --sample_step 8 --task_name "aws_realother1"

python train.py --device "cuda:2" --tau 5 --min_tau 0.5 --real_sample 4 --sample_step 16 --task_name "aws_realother2"

python train.py --device "cuda:2" --tau 5 --min_tau 0.5 --real_sample 4 --sample_step 32 --task_name "aws_realother3"

python train.py --device "cuda:2" --tau 5 --min_tau 0.5 --real_sample 4 --sample_step 64 --task_name "aws_realother4"