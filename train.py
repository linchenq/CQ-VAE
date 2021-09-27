import os
import argparse
import tqdm
import numpy as np
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import CQVAE
from utils.loss import CQVAELoss
from utils.datasets import SpineDataset
from utils.logger import Logger
from utils.summary import Summary
import utils.util as uts

    
class Trainer(object):
    def __init__(self, args, dataset, model, device):
        self.args = args
        self.model = model
        self.device = device
        self.dataloader = {
            'train': DataLoader(dataset['train'], batch_size=self.args.batch_size, shuffle=True),
            'valid': DataLoader(dataset['valid'], batch_size=self.args.batch_size, shuffle=True)
        }
        
        # DL prepared
        self.model.to(self.device)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
        self.loss = CQVAELoss(alpha=1.0, beta=1.0, gamma=1.0, device=self.device, eps=1e-20)
        
        # log and weights initialization
        self._init_folders()
        self._init_weights(load_ws=self.args.load_ws)
        self.summary = Summary(logger=Logger(self.log_pth, self.args.task_name), debug=True, loss=self.loss)
        
        # random seed
        self.random_seed = 0

    
    def _init_folders(self):
        self.log_pth = f"./logs/logs_{self.args.task_name}/"
        self.sav_pth = f"./saves/saves_{self.args.task_name}/"
        if self.args.log:
            os.makedirs(self.log_pth, exist_ok=True)
        if self.args.sav:
            os.makedirs(self.sav_pth, exist_ok=True)
    
    def _init_weights(self, load_ws):
        if load_ws is not None:
            try:
                self.model.load_state_dict(torch.load(load_ws))
            except Exception as e:
                self.summary.log("warning", repr(e))
        
        
    def train(self):
        print(f"{self.args.task_name} is under training")

        num_epoch = self.args.epoch
        for epoch in tqdm.tqdm(range(num_epoch)):
            self.run_single_step(epoch)
        
        # save model summary or log as json file
        # f"{self.sav_pth}summary_train/valid_{self.args.task_name}.json"
        modes = ['train', 'valid']
        params = [self.summary.save_train, self.summary.save_valid]
        for mode, param in zip(modes, params):
            with open(f"{self.sav_pth}summary_{mode}_{self.args.task_name}.json", 'w') as f:
                json.dump(param, fp=f, indent=4)
                
        # log for all losses throughout epoches
        log_train, log_valid = self.summary.summary_loss()
        loss_logs = [log_train, log_valid]
        for mode, log in zip(modes, loss_logs):
            with open(f"{self.sav_pth}loss_{mode}_{self.args.task_name}.txt", 'w') as f:
                f.write(log)
    
    # set lc_bool to True if the ground truth set is not enough
    # However, cfgs/gen_table.py should be modified to generate enough number of combinations
    def run_single_step(self, epoch, lc_bool=True):           
        self.model.train()
        
        epoch_loss = 0
        epoch_dict = None
        epoch_size = len(self.dataloader['train'])
        
        for batch_i, (x, meshes, best) in enumerate(self.dataloader['train']):
            x = x.to(self.device)
            best = best.to(self.device)
            meshes = meshes.to(self.device)
            
            # Update meshes: import linear combination to "extend" ground truth set
            if lc_bool:
                gts = uts._batch_lc(cfg="cfgs/cfgs_table.npy",
                                    size=self.args.gt_sample,
                                    meshes=meshes,
                                    random_seed=self.random_seed)
            else:
                gts = meshes
            
            # Network
            self.optimizer.zero_grad()
            
            output, gs_logits = self.model(x)
            loss, l_dict = self.loss.forward(output, gs_logits, gts, best, self.model.vector_dims,
                                             kld=True, best_bool=True, autoe=True, regress=True, mark=False)
            loss.backward()
            
            self.optimizer.step()
            
            # Update tau:
            #   accum_batch: The actual iteration
            #   tau_step: the number following len(self.dataloader['train_A']) is important,
            #             it represents the number of update operations
            batch_unit = len(self.dataloader['train'].dataset) // self.args.batch_size
            accum_batch = batch_unit * epoch + batch_i
            tau_step = len(self.dataloader['train']) // 3
            
            if batch_i % tau_step == 0:
                self.model.tau = np.maximum(self.model.tau * np.exp(-1e-4 * accum_batch), self.args.min_tau)
                self.summary.log("info", f"E{epoch}B{batch_i}is : {self.model.tau}")
            
            # training process log
            epoch_loss += loss.item()
            epoch_dict = uts.dict_add(epoch_dict, l_dict) if epoch_dict is not None else l_dict
            
        self.summary.model_eval(epoch, "train", epoch_loss, epoch_dict, epoch_size)
        
        if epoch % self.args.eval_step == 0:
            self.model.eval()
            self.valid(epoch)
        
        if epoch % self.args.save_step == 0:
            torch.save(self.model.state_dict(),
                        f"{self.sav_pth}ckpt_{epoch}_{self.args.task_name}.pth")
            
    def valid(self, epoch, lc_bool=True):
        self.model.eval()
        
        epoch_loss = 0
        epoch_dict = None
        epoch_size = len(self.dataloader['valid'])
        
        for batch_i, (x, meshes, best) in enumerate(self.dataloader['valid']):
            x = x.to(self.device)
            best = best.to(self.device)
            
            # Update meshes: import linear combination to "extend" ground truth set
            if lc_bool:
                meshes = [mesh.to(self.device) for mesh in meshes]
                gts = uts._batch_lc(cfg="cfgs/cfgs_table.npy",
                                    size=self.args.gt_sample,
                                    meshes=meshes,
                                    random_seed=self.random_seed)
            else:
                gts = meshes
            
            output, gs_logits = self.model(x)
            loss, l_dict = self.loss.forward(output, gs_logits, gts, best, self.model.vector_dims,
                                             kld=True, best_bool=True, autoe=True, regress=True, mark=False)

            # validation process log
            epoch_loss += loss.item()
            epoch_dict = uts.dict_add(epoch_dict, l_dict) if epoch_dict is not None else l_dict
            
        self.summary.model_eval(epoch, "valid", epoch_loss, epoch_dict, epoch_size)


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=151)
    parser.add_argument("--num_sample", type=int, default=128)
    parser.add_argument("--gt_sample", type=int, default=16)
    parser.add_argument("--task_name", type=str, default="debug")
    
    # tau: hyper-param related to gumbel softmax
    parser.add_argument("--tau", type=int, default=3)
    parser.add_argument("--min_tau", type=float, default=0.5)
    
    # log and weights related
    parser.add_argument("--eval_step", type=int, default=5)
    parser.add_argument("--save_step", type=int, default=5)
    parser.add_argument("--log", type=bool, default=True)
    parser.add_argument("--sav", type=bool, default=True)
    parser.add_argument("--load_ws", type=str, default=None)
    
    parser.add_argument("--pretrain_weights", type=str, default=None)
    # parser.add_argument("--pretrain_weights", type=str, default="./pretrain_weights/pretrain_20.pth")
    
    args = parser.parse_args()
    
    dataset = {}
    for param in ['train', 'valid']:
        dataset[param] = SpineDataset(f"dataset/{param}.txt")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = CQVAE(in_channels=1,
                  out_channels=176*2,
                  latent_dims=64,
                  vector_dims=11,
                        
                  tau=args.tau,
                  device=device,
                  num_sample=args.num_sample)
    
    trainer = Trainer(args, dataset, model, device)
    trainer.train()
    
    