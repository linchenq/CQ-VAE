import torch
import utils.util as uts


class Evaluator(object):
    def __init__(self, logger, debug):
        self.debug = debug
        self.logger = logger
        
        self.save_train = {}
        self.save_valid = {}
    
    def eval_model(self, epoch, loss, loss_dict, size, mode):
        loss /= size
        for key in loss_dict.keys():
            loss_dict[key] /= size
        index = {
            "train": self.save_train,
            "valid": self.save_valid
        }
        
        index[mode][epoch] = {"loss": "%.4f" % loss}
        self.log("info", f"{epoch}: {mode} loss is {loss}")
        self.logger.scalar_summary(f"{mode}/loss", loss, epoch)
        for key, value in loss_dict.items():
            index[mode][epoch] = {"loss": "%.4f" % loss}
            self.logger.scalar_summary(f"{mode}/{key}", value, epoch)
        
        if self.debug:
            print(f"{epoch}: {mode} loss is {loss}")
            print(uts.print_metrics(epoch, loss_dict))
    
    def eval_valid(self, epoch, model, data, device):
        model_loss = 0
        model_dict = None
        
        for batch_i, (x, meshes, best_mesh, best_mask) in enumerate(data):
            x = torch.unsqueeze(x, dim=1)
            x = x.float().to(self.device)
            meshes = [mesh.float().to(self.device) for mesh in meshes]
            best_mesh, best_mask = best_mesh.float().to(self.device), best_mask.float().to(self.device)
            
            with torch.no_grad():
                zs, decs, qy, logits, best = self.model(x, step=None)
                pts, masks = uts.batch_linear_combination(cfg="cfgs/cfgs_table.npy",
                                                          target=zs.shape[1], 
                                                          x_shape=x.shape[2:],
                                                          meshes=meshes,
                                                          best_mesh=best_mesh,
                                                          device=self.device)
    
                loss, loss_dict = self.loss.forward(zs, decs, qy, logits, best,
                                                    pts, masks,
                                                    best_mesh, best_mask,
                                                    self.model.vector_dims)
                
                model_loss += loss.item()
                if model_dict is None:
                    model_dict = loss_dict
                else:
                    model_dict = uts.dict_add(model_dict, loss_dict)
            
        self.eval_model(epoch, model_loss, model_dict, len(data), "valid")
        
    def summary_valid(self, epoch):
        metrics = uts.summary_metrics(self.save_train, self.save_valid)
        self.log("info", f"until {epoch}:\n{metrics}")
        
        if self.debug:
            print(f"until {epoch}:\n{metrics}")
    
    def log(self, hint, message):
        self.logger.log(hint, message)