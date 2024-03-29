# CQ-VAE: Coordinate Quantized VAE for Uncertainty Estimation with Application to Disk Shape Analysis from Lumbar Spine MR Images

The repository contains the source code and model from our [ICMLA 2020 paper](https://arxiv.org/abs/2010.08713)

## Updates

**[09/2021]** Update the newest version of our model CQ-VAE while set the repository to public. The [toy dataset](https://github.com/linchenq/CQ-VAE/tree/master/dataset) will be released soon.  
**[07/2020]** PCA is applied on the lumbar dataset to build a statistical shape model (SSM), which generates modes of shape variations covering about 80% of the total variations.  

## Abstract
Ambiguity is inevitable in medical images, which often results in different image interpretations (e.g. object boundaries or segmentation maps) from different human
experts. Thus, a model that learns the ambiguity and outputs a probability distribution of the target, would be valuable for
medical applications to assess the uncertainty of diagnosis. In this paper, we propose a powerful generative model to learn a
representation of ambiguity and to generate probabilistic outputs. Our model, named Coordinate Quantization Variational Autoencoder (CQ-VAE) employs a discrete latent
space with an internal discrete probability distribution by quantizing the coordinates of a continuous latent space. As a result, the output distribution from CQ-VAE is discrete. During
training, Gumbel-Softmax sampling is used to enable backpropagation through the discrete latent space. A matching algorithm is used to establish the correspondence between
model-generated samples and "ground-truth" samples, which makes a trade-off between the ability to generate new samples and the ability to represent training samples. Besides these
probabilistic components to generate possible outputs, our model has a deterministic path to output the best estimation. We demonstrated our method on a lumbar disk image dataset, and
the results show that our CQ-VAE can learn lumbar disk shape variation and uncertainty.

## Framework
![image](https://github.com/linchenq/CQ-VAE/blob/master/images/framework.jpg)

## Hyper-parameter recommendation
![image](https://github.com/linchenq/CQ-VAE/blob/master/images/training_ratio.jpg)
![image](https://github.com/linchenq/CQ-VAE/blob/master/images/eval_ratio.jpg)

## Setup

```
conda create -n pytorch1.8_py3.8_cqvae python=3.8
source activate pytorch1.8_py3.8_cqvae
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### Toy dataset preprocessing

### Training

```
python train.py [-h] [--device DEVICE] [--batch_size BATCH_SIZE] [--lr LR] [--epoch EPOCH] [--num_sample NUM_SAMPLE]
                [--gt_sample GT_SAMPLE] [--task_name TASK_NAME] [--tau TAU] [--min_tau MIN_TAU]
                [--eval_step EVAL_STEP] [--save_step SAVE_STEP] [--log LOG] [--sav SAV] [--load_ws LOAD_WS]
                [--pretrain_weights PRETRAIN_WEIGHTS]

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE
  --batch_size BATCH_SIZE
  --lr LR
  --epoch EPOCH
  --num_sample NUM_SAMPLE
  --gt_sample GT_SAMPLE
  --task_name TASK_NAME
  --tau TAU
  --min_tau MIN_TAU
  --eval_step EVAL_STEP
  --save_step SAVE_STEP
  --log LOG
  --sav SAV
  --load_ws LOAD_WS
  --pretrain_weights PRETRAIN_WEIGHTS
```

### Inference

```
python inference.py
```
