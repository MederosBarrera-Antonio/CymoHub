
# EARLY STOPPING

import os
import torch

class EarlyStopping:
    def __init__(self, 
                 patience=10, 
                 min_delta=0, 
                 verbose=False, 
                 path_base=None, 
                 min_epochs=0, 
                 mode='min',
                 loss_func=None,
                 lr_value=None,
                 scheduler_gamma = None,
                 reg_L2_value = None,
                 use_attention_gates = False):
        
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path_base = path_base
        self.best_model = None
        self.min_epochs = min_epochs
        self.mode = mode
        self.loss_func = loss_func
        self.lr_value = lr_value
        self.epoch = None
        self.last_saved_model = None
        self.scheduler_gamma = scheduler_gamma
        self.reg_L2_value = reg_L2_value
        self.use_attention_gates = use_attention_gates
        
    def __call__(self, metric, model):
        if self.epoch < self.min_epochs:
            return

        if self.best_score is None:
            self.save_checkpoint(metric, model)
            
        elif self.is_improvement(metric):
            self.save_checkpoint(metric, model)
            self.counter = 0
            
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def is_improvement(self, metric):
        if self.mode == 'min':
            return metric < self.best_score - self.min_delta
        elif self.mode == 'max':
            return metric > self.best_score + self.min_delta

    def save_checkpoint(self, metric, model):
        if self.verbose:
            if self.best_score is None:
                print(f"Saving initial model with metric: {metric:.6f}")
            else:
                print(f"Metric improved ({self.best_score:.6f} --> {metric:.6f}). Saving model ...")

        # path_ = self.path_base+self.loss_func+'_LR-'+str(self.lr)+'_ExpSchedGamma-'+str(self.scheduler_gamma)+'_L2-'+str(self.reg_L2_value)+'_Epoch-'+str(self.epoch)+"_EarlyStopping.pth"
        path_ = self.path_base+self.loss_func+'_LR-'+str(self.lr_value)+'_ExpSchedGamma-'+str(self.scheduler_gamma)+'_L2-'+str(self.reg_L2_value)+'_EPOCHS-'+str(self.epoch)+"_EarlyStopping_AG-"+str(self.use_attention_gates)+".pth"
        
        if self.last_saved_model is not None and os.path.exists(self.last_saved_model):
            os.remove(self.last_saved_model)
            if self.verbose:
                print(f"Previous model {self.last_saved_model} removed.")

        torch.save(model, path_)

        self.last_saved_model = path_

        self.best_score = metric