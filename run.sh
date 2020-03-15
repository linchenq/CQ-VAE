# cuda: 0
python train.py --device "cuda:0" --sample_step 128 --real_sample 16 --tau 2 --min_tau 0.5 --task_name "tau0"

python train.py --device "cuda:0" --sample_step 128 --real_sample 16 --tau 5 --min_tau 0.5 --task_name "tau1"

python train.py --device "cuda:0" --sample_step 128 --real_sample 16 --tau 20 --min_tau 5 --task_name "tau2"

python train.py --device "cuda:0" --sample_step 128 --real_sample 16 --tau 100 --min_tau 5 --task_name "tau3"

python train.py --device "cuda:0" --sample_step 128 --real_sample 16 --tau 0.5 --min_tau 0.5 --epoch 51 --task_name "tau4"

python train.py --device "cuda:0" --sample_step 128 --real_sample 16 --tau 2 --min_tau 2 --epoch 51 --task_name "tau5"

python train.py --device "cuda:0" --sample_step 128 --real_sample 16 --tau 5 --min_tau 5 --epoch 51 --task_name "tau6"

python train.py --device "cuda:0" --sample_step 128 --real_sample 16 --tau 50 --min_tau 50 --epoch 51 --task_name "tau7"


# cuda: 1
python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 1 --task_name "real0"

python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 4 --task_name "real1"

python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 16 --task_name "real2"

python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 32 --task_name "real3"

python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 128 --task_name "real4"

python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 256 --task_name "real5"

python train.py --device "cuda:1" --tau 5 --min_tau 0.5 --real_sample 1 --sample_step 512 --task_name "real6"

# cuda: 2
python train.py --device "cuda:2" --tau 5 --min_tau 0.5 --real_sample 8 --sample_step 32 --task_name "realother0"

python train.py --device "cuda:2" --tau 5 --min_tau 0.5 --real_sample 8 --sample_step 64 --task_name "realother1"

python train.py --device "cuda:2" --tau 5 --min_tau 0.5 --real_sample 8 --sample_step 128 --task_name "realother2"

python train.py --device "cuda:2" --tau 5 --min_tau 0.5 --real_sample 8 --sample_step 256 --task_name "realother3"

python train.py --device "cuda:2" --tau 5 --min_tau 0.5 --real_sample 8 --sample_step 512 --task_name "realother4"

# cuda: 3
python train.py --device "cuda:3" --tau 5 --min_tau 0.5 --sample_step 256 --real_sample 4 --task_name "ratio0"

python train.py --device "cuda:3" --tau 5 --min_tau 0.5 --sample_step 256 --real_sample 8 --task_name "ratio1"

python train.py --device "cuda:3" --tau 5 --min_tau 0.5 --sample_step 256 --real_sample 32 --task_name "ratio2"

python train.py --device "cuda:3" --tau 5 --min_tau 0.5 --sample_step 256 --real_sample 64 --task_name "ratio3"

python train.py --device "cuda:3" --tau 5 --min_tau 0.5 --sample_step 256 --real_sample 128 --task_name "ratio4"

python train.py --device "cuda:3" --tau 5 --min_tau 0.5 --sample_step 256 --real_sample 256 --task_name "ratio5"
