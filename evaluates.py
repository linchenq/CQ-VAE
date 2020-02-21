import torch
import utils.util as uts


class Evaluator(object):
    def __init__(self, logger, debug):
        self.debug = debug
        self.logger = logger
        
        self.train_loss, valid_loss = {}, {}
    
    def train_eval(self, epoch, loss, size):
        loss /= size
        self.train_loss[epoch] = loss
        
        self.log("info", f"{epoch}: train loss is {loss}")
        self.logger.scalar_summary("train/loss", loss, epoch)
        if self.debug:
            print(f"{epoch}: train loss is {loss}")
        
    def valid_eval(self, epoch, model, data, device, filename):
        loss = 0
        for batch_i, (x, meshes) in enumerate(data):
            x = torch.unsqueeze(x, dim=1)
            x = x.float().to(device)
            meshes = [mesh.float().to(device) for mesh in meshes]
            
            with torch.no_grad():
                zs, decs, qy, logits, best = model(x, step=None)
                pts, masks = uts.batch_linear_combination(cfg="cfgs/cfgs_table.npy",
                                                          target=zs.shape[1], 
                                                          x_shape=x.shape[2:], meshes=meshes,
                                                          device=device)
                # valid_loss = inference
                # loss += valid_loss.item()
        loss /= len(data)
        self.valid_loss[epoch] = loss
        
        self.log("info", f"{epoch}: valid loss is {loss}")
        self.logger.scalar_summary("valid/loss", loss, epoch)
        
        if self.debug:
            print(f"{epoch}: valid loss is {loss}")
        
    def metric_eval(self, epoch, filename):
        metrics = uts.print_metrics(self.train_dict, self.valid_dict)
        self.log("info", f"util {epoch}:\n{metrics}")
        uts.plot_loss(epoch, self.train_dict, self.valid_dict, filename)
        
        if self.debug:
            print(f"util {epoch}:\n{metrics}")
    
    def log(self, hint, message):
        self.logger.log(hint, message)