from terminaltables import AsciiTable

class Summary(object):
    def __init__(self, logger, debug, loss):
        self.debug = debug
        self.logger = logger

        self.loss = loss
    
        self.save_train = {}
        self.save_valid = {}
        
    def log(self, hint, message):
        self.logger.log(hint, message)
        
    def model_eval(self, epoch, mode, losses, dicts, size):
        loss = losses / size
        
        # log
        self.log("info", f"{epoch}: {mode} loss is {loss}")
        self.logger.scalar_summary(f"{mode}/loss", loss, epoch)
        
        # convert tensors to normal dict
        ret = {'loss': loss}
        for k, v in dicts.items():
            ret[f"{k}"] = v / size
            self.logger.scalar_summary(f"{mode}/{k}", v, epoch)
        
        if mode == 'train':
            self.save_train[epoch] = ret
        elif mode == 'valid':
            self.save_valid[epoch] = ret
        else:
            raise NotImplementedError
            
        # print in command with debug MODE
        print(f"{epoch}: {mode} loss is {loss}")
        if self.debug:
            print(self.__print_metrics(epoch, ret))
            
    def summary_loss(self):
        return self.__print_summary(self.save_train), self.__print_summary(self.save_valid)
    
    
    def __print_metrics(self, epoch, dicts):
        # main keys: 'loss', ...
        metrics = [['Epoch'] + list(dicts.keys())]
        met_v = ["%.2f" % i for i in dicts.values()]
        values = [str(epoch)] + [str(i) for i in met_v]
        
        metrics.append(values)
        
        return AsciiTable(metrics).table
    
    def __print_summary(self, dicts):
        metrics = [['Epoch'] + list(dicts[0].keys())]
        for epoch in dicts.keys():
            row = [str(epoch)]
            for main_key in dicts[epoch].keys():
                row.append(dicts[epoch][main_key])
            metrics.append(row)
            
        return AsciiTable(metrics).table