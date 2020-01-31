import os
import argparse
import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import utils.utils as uts
from model import DiscreteVAE
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
                self.logger.log("WAR", "Unknown weights file")
                raise NotImplementedError
        
        # DL prepared
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.args.lr)
        
        # loss recorder
        self.train_dict = {}
        self.valid_dict = {}
        
    def train(self):
        num_epoch = self.args.epoch
        
        for epoch in tqdm.tqdm(range(num_epoch)):
            self.run_single_step(epoch)
    
    def evaluate(self, epoch):
        print("evaluating...")
        valid_loss = 0
        for batch_i, (x, mesh) in enumerate(self.dataloader['valid']):
            x, mesh = x.float(), mesh.float()
            x, mesh = x.to(self.device), mesh.to(self.device)
            
            with torch.no_grad():
                pts, qr = self.model(x)
                loss = self.model.loss(pts, mesh, qr)
            
            valid_loss += loss.item()
        valid_loss /= len(self.dataloader['valid'])
        self.valid_dict[epoch] = valid_loss
        
        print(f"{epoch}: valid loss is {valid_loss}")
        self.logger.log("INF", f"{epoch}: valid loss is {valid_loss}")
        self.logger.scalar_summary("valid/loss", valid_loss, epoch)
        
    def eval_record(self, epoch):
        metrics = uts.print_metrics(self.train_dict, self.valid_dict)
        print(f"util {epoch}:\n{metrics}")
        self.logger.log("INF", f"util {epoch}:\n{metrics}")
        
        uts.plot_loss(epoch, self.train_dict, self.valid_dict)
        
    
    def run_single_step(self, epoch):
        self.model.train()
        train_loss = 0
        
        for batch_i, (x, mesh) in enumerate(self.dataloader['train']):
            x, mesh = x.float(), mesh.float()
            x, mesh = x.to(self.device), mesh.to(self.device)
            self.optimizer.zero_grad()
            
            with torch.autograd.detect_anomaly():
                pts, qy = self.model(x)
                loss = self.model.loss(pts, mesh, qy)
                loss.backward()
                self.optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(self.dataloader['train'])
        self.train_dict[epoch] = train_loss
        
        print(f"{epoch}: train loss is {train_loss}")
        self.logger.log("INF", f"{epoch}: train loss is {train_loss}")
        self.logger.scalar_summary("train/loss", train_loss, epoch)
        
        if epoch % self.args.eval_step == 0:
            self.evaluate(epoch)
            self.eval_record(epoch)
        
        if epoch % self.args.save_step == 0:
            torch.save(self.model.state_dict(), f"{self.args.sav_pth}ckpt_{self.args.task_name}_{epoch}.pth")


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=11)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--task_name", type=str, default="test")
    
    parser.add_argument("--eval_step", type=int, default=5)
    parser.add_argument("--save_step", type=int, default=5)
    
    parser.add_argument("--log_pth", type=str, default="./logs/")
    parser.add_argument("--sav_pth", type=str, default="./saves/")
    parser.add_argument("--pretrained_weights", type=str, default=None)
    
    args = parser.parse_args()
    
    dataset = {}
    for param in ['train', 'valid']:
        dataset[param] = SpineDataset(f"dataset/{param}.txt")
    model = DiscreteVAE(in_ch=1, out_ch=176*2, 
                        latent_dims=64, vec_dims=11, 
                        beta=1., temperature=1., 
                        device=args.device)
    
    trainer = Trainer(args, dataset, model)
    trainer.train()
    
    