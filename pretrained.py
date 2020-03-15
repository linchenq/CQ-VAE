import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from utils.ops import RegressAutoEncoder, DiscreteDecoder
from torch.utils.data import DataLoader

from evals.evaluates import Evaluator
from utils.datasets import SpineDataset


class Pretrained(nn.Module):
    def __init__(self,
                 out_channel=176*2,
                 latent_dims=64,
                 vector_dims=11):
        
        super(Pretrained, self).__init__()
        
        self.autoencoder = RegressAutoEncoder(out_channel, latent_dims, vector_dims)
        self.decoder = DiscreteDecoder((latent_dims * vector_dims), out_channel, 1)

    def forward(self, x):
        x = x.view(-1, 176*2)
        z = self.autoencoder(x)
        z = z.view(-1, 64*11)
        rx, _ = self.decoder(z)
        rx = rx.view(-1, 176, 2)
        return rx
    
    def model_loss(self, rx, x):
        loss = F.mse_loss(rx, x)
        return loss

def pretrain(num_epoch, lr, device):
    model = Pretrained(out_channel=176*2,
                       latent_dims=64,
                       vector_dims=11)
    # model.load_state_dict(torch.load("./pretrain_weights/pretrain_20.pth"))
    
    model = model.to(device)
    
    dataset = {}
    for param in ['train', 'valid']:
        dataset[param] = SpineDataset(f"dataset/{param}.txt")
    
    dataloader = {
        'train': DataLoader(dataset['train'], batch_size=1, shuffle=True),
        'valid': DataLoader(dataset['valid'], batch_size=1, shuffle=True)
    }
    
    for epoch in tqdm.tqdm(range(num_epoch)):
        model.train()
        
        epoch_loss = 0
        epoch_size = 0
        
        for batch_i, (x, meshes, _, _) in enumerate(dataloader['train']):
            meshes = torch.squeeze(meshes, dim=0)
            meshes = meshes.float().to(device)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=lr)
            optimizer.zero_grad()
            
            rx = model(meshes)
            model_loss = model.model_loss(rx, meshes)
            model_loss.backward()
            
            optimizer.step()
            
            epoch_loss += model_loss.item() * meshes.shape[0]
            epoch_size += meshes.shape[0]
        
        print(f"{epoch} loss is {epoch_loss/epoch_size}")
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                        f"pretrain_weights/pretrain_{epoch}.pth")


if __name__ == '__main__':
    # pretrain(num_epoch=21, lr=1e-5, device="cuda:0")
    