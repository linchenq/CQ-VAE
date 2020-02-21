import os
import argparse
import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import utils.util as uts
from model import DiscreteVAE
from evaluates import Evaluator
from utils.loss import DiscreteLoss
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
        os.makedirs(self.args.log_pth, exist_ok=True)
        os.makedirs(self.args.sav_pth, exist_ok=True)
        
        # Eval
        self.evaluator = Evaluator(logger=Logger(self.args.log_pth, self.args.task_name), debug=True)
        
        # pretrained weights loading
        if self.args.pretrained_weights is not None:
            if self.args.pretrained_weights.endswith(".pth"):
                self.model.load_state_dict(torch.load(self.args.pretrained_weights))
            else:
                self.evaluator.log("warning", "Unknown weight files: Error at train.py")
                raise NotImplementedError("Unknown weight files: Error at train.py")
        
        # DL prepared
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.args.lr)
        self.loss = DiscreteLoss(alpha=1.0, beta=1.0, gamma=1.0, device=self.device, eps=1e-20)
        
    def train(self):
        print(f"{self.args.task_name} is under training")

        num_epoch = self.args.epoch
        for epoch in tqdm.tqdm(range(num_epoch)):
            self.run_single_step(epoch)
    
    def run_single_step(self, epoch):
        self.model.train()
        
        epoch_loss = 0
        for batch_i, (x, meshes) in enumerate(self.dataloader['train']):
            x = torch.unsqueeze(x, dim=1)
            x = x.float().to(self.device)
            meshes = [mesh.float().to(self.device) for mesh in meshes]
            self.optimizer.zero_grad()
            
            zs, decs, qy, logits, best = self.model(x, step=None)
            pts, masks = uts.batch_linear_combination(cfg="cfgs/cfgs_table.npy",
                                                      target=zs.shape[1], 
                                                      x_shape=x.shape[2:], meshes=meshes,
                                                      device=self.device)
                
            # model_loss = self.loss.forward(zs, decs, qy, logits, best,
            #                                pts, masks, )
            # loss.backward()
            
            self.optimizer.step()
            
            # epoch_loss += loss.item()
            
        # self.evaluator.train_eval(epoch, epoch_loss, len(self.dataloader['train']))
        
        #     # optimize on tau
        #     if batch_i % batch_step == 0:
        #         model.tau = np.maximum(model.tau * np.exp(-3e-5 * batch_i),
        #                                 0.5)
        #         self.logger.log("info", f"E{epoch}B{batch_i}is : {model.tau}")
            
        #     train_loss += loss.item()
        
        # if epoch % self.args.eval_step == 0:
        #     self.evaluate(epoch)
        #     self.eval_record(epoch, f"{self.args.sav_pth}{self.args.task_name}_{self.args.task_name}_{epoch}.jpg")
        
        # if epoch % self.args.save_step == 0:
        #     torch.save(self.model.state_dict(), f"{self.args.sav_pth}ckpt_{self.args.task_name}_{epoch}.pth")


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task_name", type=str, default="debug_task")
    parser.add_argument("--sample_step", type=int, default=128)
    
    parser.add_argument("--eval_step", type=int, default=2)
    parser.add_argument("--save_step", type=int, default=2)
    
    parser.add_argument("--log_pth", type=str, default="./logs/")
    parser.add_argument("--sav_pth", type=str, default="./saves/")
    parser.add_argument("--pretrained_weights", type=str, default=None)
    
    args = parser.parse_args()
    
    dataset = {}
    for param in ['train', 'valid']:
        dataset[param] = SpineDataset(f"dataset/{param}.txt")
        
    model = DiscreteVAE(in_channels=1, out_channels=176*2, seg_channels=1,
                        latent_dims=64, vector_dims=11, 
                        alpha=1., beta=1., tau=1., 
                        device=args.device)
    
    trainer = Trainer(args, dataset, model)
    trainer.train()
    
    