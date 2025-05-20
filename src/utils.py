import os
import random
import numpy as np
import torch
from datetime import datetime

def set_seed(seed=42,logger=print):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger(f'random seed with {seed}')

class Logger:
    def __init__(self, model_name):
        self.model_name=model_name
        self.date=str(datetime.now().date()).replace('-','')[2:]
        import os
        if not os.path.exists('log'):
            os.makedirs('log')
        self.logger_file = f'log/{self.date}_{self.model_name}'
        
    def __call__(self, text, verbose=True, log=True):
        if log:
            with open(f'{self.logger_file}.log', 'a') as f:
                f.write(f'[{datetime.now().replace(microsecond=0)}] {text}\n')
        if verbose:
            print(f'[{datetime.now().time().replace(microsecond=0)}] {text}')

class EarlyStopper:
    def __init__(self, patience=7, printfunc=print, verbose=True, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): epochs to wait after minimum has been reached.
                            Default: 7
            verbose (bool): whether to print the early stopping message or not.
                            Default: False
            delta (float): minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): path to save the model when validation loss decreases.
                            Default: 'checkpoint.pt'
            printfunc (func): print function to use.
                            Default: python print function
        """
        self.printfunc=printfunc
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score * (1+ self.delta):
            self.counter += 1
            if self.verbose:
                self.printfunc(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        if self.early_stop:
            return True

    def save_checkpoint(self, val_loss, model):
        ''' Saves model when validation loss decrease. '''
        # if self.verbose:
        #     self.printfunc(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        import os
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss