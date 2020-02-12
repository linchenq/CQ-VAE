#!/bin/bash

python train.py --epoch 100 --task_name "E100@B2" --batch_step 2

python train.py --epoch 100 --task_name "E100@B4" --batch_step 4

python train.py --epoch 100 --task_name "E100@B6" --batch_step 6

python train.py --epoch 100 --task_name "E100@B8" --batch_step 8

python train.py --epoch 100 --task_name "E100@B10" --batch_step 10

python train.py --epoch 100 --task_name "E100@B12" --batch_step 12

python train.py --epoch 100 --task_name "E100@B14" --batch_step 14

python train.py --epoch 100 --task_name "E100@B16" --batch_step 16

python train.py --epoch 100 --task_name "E100@B18" --batch_step 18

python train.py --epoch 100 --task_name "E100@B20" --batch_step 20

python train.py --epoch 100 --task_name "E100@B25" --batch_step 25

python train.py --epoch 100 --task_name "E100@B30" --batch_step 30

python train.py --epoch 100 --task_name "E100@B50" --batch_step 50

python train.py --epoch 100 --task_name "E100@B70" --batch_step 70

python train.py --epoch 100 --task_name "E100@B100" --batch_step 100

python train.py --epoch 100 --task_name "E100@B200" --batch_step 200

python train.py --epoch 150 --task_name "BATCH2" --batch_step 5 --batch_size 2

python train.py --epoch 150 --task_name "BATCH3" --batch_step 5 --batch_size 3

python train.py --epoch 150 --task_name "BATCH5" --batch_step 5 --batch_size 5

python train.py --epoch 150 --task_name "BATCH10" --batch_step 5 --batch_size 10

python train.py --epoch 150 --task_name "BATCH20" --batch_step 5 --batch_size 20

python train.py --epoch 150 --task_name "BATCH50" --batch_step 5 --batch_size 50

python train.py --epoch 150 --task_name "BATCH100" --batch_step 5 --batch_size 100


python train.py --epoch 150 --task_name "LR1E3" --lr 1e-3 --batch_step 10

python train.py --epoch 150 --task_name "LR5E4" --lr 5e-4 --batch_step 10

python train.py --epoch 150 --task_name "LR1E4" --lr 1e-4 --batch_step 10

python train.py --epoch 150 --task_name "LR5E3" --lr 5e-3 --batch_step 10


