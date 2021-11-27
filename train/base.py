import abc
import numpy as np
import os

from utils.functions import wLoss
from .model import BaseModel
from utils.setting import crema_setting, get_model_dir, get_pickle_path
import random
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch.optim as optim
import torch
import random
from enum import Enum

class DataType(Enum):
    X_TRAIN = 1
    X_TEST = 2
    X_VALIDATION = 3
    Y_TRAIN = 4
    Y_TEST = 5
    Y_VALIDATION = 6
    YS_TEST = 7
    
class ModelRunner():
    def __init__(self,dataset):
        self.init_setting()
        self.dataset = dataset
        self.model_dir = get_model_dir(self.dataset)
        self.data_path = get_pickle_path(self.dataset)
        self.dataloader = {}

    def init_setting(self):
        self.setting = crema_setting

    def set_random_seed(self,seed):
        print('SEED %d start' %(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @abc.abstractmethod
    def get_n_fold(self):
        pass

    def init_network(self):
        self.model = BaseModel().cuda()
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.setting.lr)
        self.my_loss   = wLoss().cuda() #FocalLoss().cuda() #
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='max', patience=2) 
   
    @abc.abstractmethod
    def set_data(self,fold):
        pass

    def run(self): 
        UA_valid, UA_test = ([] for _ in range(2))

        n_fold = self.get_n_fold()
        UA_valid.append([]), UA_test.append([])

        for fold in range(n_fold):
            fold_UA_valid, fold_UA_test = self.run_fold(fold)
            UA_valid[-1] += fold_UA_valid
            UA_test[-1] += fold_UA_test
        
        print("WC Domain [%s] valid UA: %.2f-%.4f test UA %.2f-%.4f" %(self.dataset, np.mean(UA_valid[-1]),np.std(UA_valid[-1]),
                np.mean(UA_test[-1]),np.std(UA_test[-1])))
        return UA_valid[-1],UA_test[-1]

    def run_fold(self,fold):
        ls_train,n_minibatch,tr_n_samples = self.set_data(fold)
        fold_UA_valid,fold_UA_test = ([] for _ in range(2))

        for seed in range(self.setting.n_seeds):
            best_UA_valid = 0.
            best_UA_test  = 0.
            best_UA_valid, best_UA_test = self.run_seed(seed,ls_train,tr_n_samples,n_minibatch,fold)
            fold_UA_valid.append([best_UA_valid])
            fold_UA_test.append([best_UA_test])
        return fold_UA_valid, fold_UA_test

    @abc.abstractmethod
    def run_seed(self,seed,ls_train,tr_n_samples,n_minibatch,fold):
        pass

