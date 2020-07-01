import torch

from utils.loss import CQLoss
import utils.util as uts

class Evaluator(object):
    def __init__(self, logger, debug, cfg):
        self.debug = debug
        self.logger = logger
        self.cfg = cfg
        
        self.random_seed = 0
        self.loss = None
    
        self.save_train = {}
        self.save_valid = {}
        
    
    def eval_model(self, epoch, mode, loss, loss_dict, size):
        # log
        self.log("info", f"{epoch}: {mode} loss is {loss}")
        self.logger.scalar_summary(f"{mode}/loss", loss, epoch)
        
        # accumlated loss should be divided by its size
        for k in loss_dict.keys():
            loss_dict[k] /= size
        loss /= size
                
        # combine loss in one dictionary
        ret = {"loss": "%.4f" % loss}
        for k, v in loss_dict.items():
            ret[f"{k}"] = "%.4f" % v
            self.logger.scalar_summary(f"{mode}/{k}", v, epoch)
            
        # save in global dictionary
        if mode == "train":
            self.save_train[epoch] = ret
        elif mode == "valid":
            self.save_valid[epoch] = ret
        else:
            raise NotImplementedError

        # print in command
        if self.debug:
            print(f"{epoch}: {mode} loss is {loss}")
            print(uts.print_metrics(epoch, loss_dict))
    
    
    def eval_valid(self, epoch, model, data, real_sample, device):
        model.eval()
        
        self.loss = CQLoss(alpha=model.alpha, beta=model.beta, gamma=model.gamma, device=device, eps=1e-20)
        
        e_loss, e_size, e_dict = 0, 0, None
        
        for batch_i, (x, meshes, best_mesh) in enumerate(data):
            x = torch.unsqueeze(x, dim=1)
            x, best_mesh = x.float().to(device), best_mesh.float().to(device)
            meshes = [mesh.float().to(device) for mesh in meshes]
            
            with torch.no_grad():
                # zs, decs, best are generated from model, called sampling ones
                zs, decs, qy, logits, best = model(x)
            
                # pts, masks are linear combination of ground truth samples
                # The number of generated ground truth samples is self.args.real_sample
                # Please ensure self.args.real_sample <= self.num_sample
                pts = uts.batch_linear_combination(cfg=self.cfg,
                                                   target=real_sample, 
                                                   meshes=meshes,
                                                   random_seed=self.random_seed)
                loss, loss_dict = self.loss.forward(zs, decs, qy, logits, best,
                                                    pts, best_mesh, model.vector_dims)
                
                e_loss += loss.item() * x.shape[0]
                e_size += x.shape[0]
                if e_dict is None:
                    e_dict = loss_dict
                else:
                    e_dict = uts.dict_add(e_dict, loss_dict)
            
        self.eval_model(epoch, "valid", e_loss, e_dict, e_size)
    

    def update_seed(self, random_seed):
        self.random_seed = random_seed
        
        
    def summary_valid(self, epoch):
        metrics = uts.summary_metrics(self.save_train, self.save_valid)
        self.log("info", f"until {epoch}:\n{metrics}")
        
        if self.debug:
            print(f"until {epoch}:\n{metrics}")
    
    
    def log(self, hint, message):
        self.logger.log(hint, message)