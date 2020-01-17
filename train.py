import os
import argparse
import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.logger import Logger

class Trainer(object):
    def __init__(self, args, dataset, model):
        self.args = args
        self.model = model
        
        self.dataloader = {
            'train': DataLoader(dataset['train'], batch_size=self.args.batch_size, shuffle=True),
            'valid': DataLoader(dataset['valid'], batch_size=self.args.batch_size, shuffle=True)
        }
        
        # prerequistites
        os.makedirs(self.args.log_pth, exist_ok=True)
        os.makedirs(self.args.sav_pth, exist_ok=True)
        
        # logger and pretrained weights
        self.logger = Logger()
        if self.args.pretrained_weights is not None:
            if self.args.pretrained_weights.endswith(".pth"):
                self.model.load_state_dict(torch.load(self.args.pretrained_weights))
            else:
                self.logger.log_summary(mode="WARNING", msg="Unknown pretrained weights or models")

        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.args.lr)
        
    def train(self):
        num_epoch = self.args.epoch
        
        for epoch in tqdm.tqdm(range(num_epoch)):
            self.run_single_step(epoch)
    
    def run_single_step(self, epoch):
        self.model.train()
        
        for batch_i, (image, mesh) in enumerate(self.dataloader['train']):
            image, mesh = image.to(self.device), mesh.to(self.device)
            self.optimizer.zero_grad()
            
            with torch.enable_grad():
                xr, mu, logvar = self.model(image)
                loss = self.model.loss(image, xr, mu, logvar)
                loss.backward()
                self.optimizer.step()
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    parser.add_argument("--log_pth", type=str, default="./logs/")
    parser.add_argument("--sav_pth", type=str, default="./saves/")
    parser.add_argument("--pretrained_weights", type=str, default=None)
    