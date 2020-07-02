import os
import argparse
import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
    
import utils.util as uts
from model import CQVAE
from evals.evaluates import Evaluator
from utils.loss import CQLoss
from utils.datasets import SpineDataset
from utils.logger import Logger
    
class Trainer(object):
    def __init__(self, args, dataset, model):
        self.args = args
        self.model = model
        self.dataloader = {
            'train': DataLoader(dataset['train'], batch_size=self.args.batch_size, shuffle=True),
            'valid': DataLoader(dataset['valid'], batch_size=self.args.batch_size, shuffle=True)
        }
        
        # prerequistites and log initial
        self.init_folders()
        
        # Eval
        self.evaluator = Evaluator(logger=Logger(self.args.log_pth, self.args.task_name), debug=True, cfg="./cfgs/cfgs_table.npy")
        
        # loading weights and pretrained weights initialization
        self.init_weights(load_weight=self.args.load_weights,
                          pretrain_weight=self.args.pretrain_weights,
                          freeze=True)
        
        # DL prepared
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.args.lr)
        self.loss = CQLoss(alpha=model.alpha, beta=model.beta, gamma=model.gamma, device=self.device, eps=1e-20)
        
        # random seed
        self.random_seed = 0
        
    
    def init_folders(self):
        # Trick Modification for multiple processes
        if self.args.log_pth == "./logs/":
            self.args.log_pth = f"./logs_{self.args.task_name}/"
        if self.args.sav_pth == "./saves/":
            self.args.sav_pth = f"./saves_{self.args.task_name}/"
            
        os.makedirs(self.args.log_pth, exist_ok=True)
        os.makedirs(self.args.sav_pth, exist_ok=True)
    
    def init_weights(self, load_weight, pretrain_weight, freeze=True):
        if load_weight is not None:
            if load_weight.endswith(".pth"):
                self.model.load_state_dict(torch.load(load_weight))
            else:
                self.evaluator.log("warning", "Unknown loaded weight files: Error at train.py")
                raise NotImplementedError("Unknown loaded weight files: Error at train.py")
        '''
           LAYERS      UPDATE_PRETRAIN        FREEZE
        autoencoder          YES                YES
        regress dec          YES                YES
        segment dec          NO                 NO
        shared pre           YES                NO     
        '''
        if pretrain_weight is not None:
            model_dict = self.model.state_dict()
            update_dict = {}
            for pre_k, pre_v in torch.load(pretrain_weight).items():
                if pre_k.startswith("autoencoder") or pre_k.startswith("decoder.regress"):
                    update_dict[pre_k] = pre_v
                elif pre_k.startswith("decoder.segment"):
                    pass
                else:
                    update_dict[pre_k] = pre_v
            model_dict.update(update_dict)
            self.model.load_state_dict(model_dict)
            
            if freeze:
                # requires_grad freeze
                for p in self.model.autoencoder.parameters():
                    p.requires_grad = False
                for p in self.model.decoder.regress.parameters():
                    p.requires_grad = False
        
        
    def train(self):
        print(f"{self.args.task_name} is under training")

        num_epoch = self.args.epoch
        for epoch in tqdm.tqdm(range(num_epoch)):
            self.run_single_step(epoch)
    
    def run_single_step(self, epoch):           
        self.model.train()
        
        # initial metrics for training and evaluating
        batch_step_tau = len(self.dataloader['train']) // self.args.batch_step_tau
        
        e_loss, e_size, e_dict = 0, 0, None
        
        for batch_i, (x, meshes, best_mesh) in tqdm.tqdm(enumerate(self.dataloader['train'])):
            x = torch.unsqueeze(x, dim=1)
            x, best_mesh = x.float().to(self.device), best_mesh.float().to(self.device)
            meshes = [mesh.float().to(self.device) for mesh in meshes]
            
            self.optimizer.zero_grad()
            
            # zs, decs, best are generated from model, called sampling ones
            zs, decs, qy, logits, best = self.model(x)
            
            # pts, masks are linear combination of ground truth samples
            # The number of generated ground truth samples is self.args.real_sample
            # Please ensure self.args.real_sample <= self.num_sample
            pts = uts.batch_linear_combination(cfg="cfgs/cfgs_table.npy",
                                               target=self.args.real_sample, 
                                               meshes=meshes,
                                               random_seed=self.random_seed)
            loss, loss_dict = self.loss.forward(zs, decs, qy, logits, best,
                                                pts, best_mesh, self.model.vector_dims)
            loss.backward()
            
            self.optimizer.step()
            
            # Update random seed, ensure dataset are selected periodically similiar
            # Update tau to let it go smaller
            if batch_i % batch_step_tau == 0:
                model.tau = np.maximum(model.tau * np.exp(-1e-4 * batch_i),
                                        self.args.min_tau)
                self.evaluator.log("info", f"E{epoch}B{batch_i}is : {model.tau}")
                
                self.random_seed += 1
                self.evaluator.update_seed(self.random_seed)
            
            # evaluation
            e_loss += loss.item() * x.shape[0]
            e_size += x.shape[0]
            if e_dict is None:
                e_dict = loss_dict
            else:
                e_dict = uts.dict_add(e_dict, loss_dict)
        
        self.evaluator.eval_model(epoch, "train", e_loss, e_dict, e_size)

        if epoch % self.args.eval_step == 0:
            self.model.eval()
            
            self.evaluator.eval_valid(epoch=epoch,
                                      model=self.model,
                                      data=self.dataloader['valid'],
                                      real_sample=self.args.real_sample,
                                      device=self.device)
            self.evaluator.summary_valid(epoch)
        
        if epoch % self.args.save_step == 0:
            torch.save(self.model.state_dict(),
                        f"{self.args.sav_pth}ckpt_{self.args.task_name}_{epoch}.pth")


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=101)
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task_name", type=str, default="db")
    
    parser.add_argument("--num_sample", type=int, default=256)
    parser.add_argument("--real_sample", type=int, default=32)
    parser.add_argument("--batch_step_tau", type=int, default=5)
    
    parser.add_argument("--tau", type=int, default=5)
    parser.add_argument("--min_tau", type=float, default=0.5)
    
    parser.add_argument("--eval_step", type=int, default=10)
    parser.add_argument("--save_step", type=int, default=10)
    
    parser.add_argument("--log_pth", type=str, default="./logs/")
    parser.add_argument("--sav_pth", type=str, default="./saves/")
    parser.add_argument("--load_weights", type=str, default=None)
    
    parser.add_argument("--pretrain_weights", type=str, default=None)
    # parser.add_argument("--pretrain_weights", type=str, default="./pretrain_weights/pretrain_20.pth")
    
    args = parser.parse_args()
    
    dataset = {}
    for param in ['train', 'valid']:
        dataset[param] = SpineDataset(f"dataset/{param}.txt")

    model = CQVAE(in_channels=1,
                  out_channels=176*2,
                  latent_dims=64,
                  vector_dims=11,
                        
                  alpha=1.,
                  beta=1.,
                  gamma=1.,
                        
                  tau=args.tau,
                  device=args.device,
                  num_sample=args.num_sample)
    
    trainer = Trainer(args, dataset, model)
    trainer.train()
    
    