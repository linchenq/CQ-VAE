import os
import argparse
import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import utils.utils as uts
from model import DiscreteVAE
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
        
        self.logger = Logger(self.args.log_pth, self.args.task_name)
        if self.args.pretrained_weights is not None:
            if self.args.pretrained_weights.endswith(".pth"):
                self.model.load_state_dict(torch.load(self.args.pretrained_weights))
            else:
                self.logger.log("warn", "Unknown weights file")
                raise NotImplementedError
        
        # DL prepared
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model_loss = DiscreteLoss(alpha=self.model.alpha, beta=self.model.beta, device=self.device)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.args.lr)
        
        # loss recorder
        self.train_dict = {}
        self.valid_dict = {}
        
    def train(self):
        print(f"{self.args.task_name} is under training")
        num_epoch = self.args.epoch
        
        for epoch in tqdm.tqdm(range(num_epoch)):
            self.run_single_step(epoch)
            
        # record for debugger
        self.logger.log("info", self.train_dict.keys())
        self.logger.log("info", self.train_dict.values())
        self.logger.log("info", self.valid_dict.keys())
        self.logger.log("info", self.valid_dict.values())
    
    def run_single_step(self, epoch):
        self.model.train()
        train_loss = 0
        batch_step = len(self.dataloader['train']) // (self.args.batch_size * self.args.batch_step)
        
        for batch_i, (x, gt_mask, gt_pts) in enumerate(self.dataloader['train']):
            x, gt_mask, gt_pts = x.float(), gt_mask.float(), gt_pts.float()
            x, gt_mask, gt_pts = x.to(self.device), gt_mask.to(self.device), gt_pts.to(self.device)
            self.optimizer.zero_grad()
            
            pts, mask, qy = self.model(x)
            loss = self.model_loss.forward(pts, gt_pts,
                                            mask, gt_mask,
                                            qy, self.model.vector_dims)
            loss.backward()
            self.optimizer.step()

            # optimize on tau
            if batch_i % batch_step == 0:
                model.tau = np.maximum(model.tau * np.exp(-3e-5 * batch_i),
                                        0.5)
                self.logger.log("info", f"E{epoch}B{batch_i}is : {model.tau}")
            
            train_loss += loss.item()
        
        train_loss /= len(self.dataloader['train'])
        self.train_dict[epoch] = train_loss
        
        print(f"{epoch}: train loss is {train_loss}")
        self.logger.log("info", f"{epoch}: train loss is {train_loss}")
        self.logger.scalar_summary("train/loss", train_loss, epoch)
        
        if epoch % self.args.eval_step == 0:
            self.evaluate(epoch)
            self.eval_record(epoch, f"{self.args.sav_pth}{self.args.task_name}_{self.args.task_name}_{epoch}.jpg")
        
        if epoch % self.args.save_step == 0:
            torch.save(self.model.state_dict(), f"{self.args.sav_pth}ckpt_{self.args.task_name}_{epoch}.pth")

    def evaluate(self, epoch):
        print("evaluating...")
        valid_loss = 0
        for batch_i, (x, gt_mask, gt_pts) in enumerate(self.dataloader['valid']):
            x, gt_mask, gt_pts = x.float(), gt_mask.float(), gt_pts.float()
            x, gt_mask, gt_pts = x.to(self.device), gt_mask.to(self.device), gt_pts.to(self.device)
            
            with torch.no_grad():
                pts, mask, qr = self.model(x)
                loss = self.model_loss.forward(pts, gt_pts,
                                               mask, gt_mask,
                                               qr, self.model.vector_dims)
            
            valid_loss += loss.item()
        valid_loss /= len(self.dataloader['valid'])
        self.valid_dict[epoch] = valid_loss
        
        print(f"{epoch}: valid loss is {valid_loss}")
        self.logger.log("info", f"{epoch}: valid loss is {valid_loss}")
        self.logger.scalar_summary("valid/loss", valid_loss, epoch)
        
    def eval_record(self, epoch, filename=None):
        metrics = uts.print_metrics(self.train_dict, self.valid_dict)
        print(f"util {epoch}:\n{metrics}")
        self.logger.log("info", f"util {epoch}:\n{metrics}")
        
        uts.plot_loss(epoch, self.train_dict, self.valid_dict, filename)


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task_name", type=str, default="debug_task")
    parser.add_argument("--batch_step", type=int, default=6)
    
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
    
    