import os
import pickle

import torch
from utils.functions import makedirs, normalization_ops, wLoss
import numpy as np
from tqdm import tqdm
from utils.setting import get_dataset_folder, get_model_dir, device, get_pickle_path

class FgsmPickleMaker:
    def __init__(self,dataset,type=None,value=None):
        self.dataset:str = dataset
        self.data_path:str = get_pickle_path(dataset,type,value)
        self.model_dir:str = get_model_dir(dataset,type,value)


    def __load_dataset(self):
        train_filename = f"{self.data_path}.pickle"
    
        with open(train_filename, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def __load_model(self,fold,seed):
        models = []
        fold = 0
        while True:
            seed = 0
            while True:
                model_path = os.path.join(self.model_dir,f"WC_fold{fold}_seed{seed}.pth")
                if os.path.exists(model_path) == False:
                    break
                my_net:torch.nn.Module = torch.load(model_path)
                my_net.eval()
                models.append(my_net)
                seed += 1
            if seed == 0:
                break
            fold += 1
        return models
        
    def fsgm(self,source:np.ndarray,data_grads:torch.Tensor,epsilon:float):
        fsgm_sources = []
        for data_grad in data_grads:
            sign_data_grad = data_grad.sign().cpu().numpy()
            fsgm_source = source + epsilon*sign_data_grad
            fsgm_sources.append(fsgm_source)
        return fsgm_sources

    def get_model_result(self,models,x_data,y_data,ls_train):
        x_eval = torch.Tensor(x_data).to(device).cuda().unsqueeze(0)
        x_eval = x_eval
        x_eval.requires_grad = True
        y_eval = torch.Tensor([y_data]).to(device).long().cuda()
        ls_train_batch = torch.Tensor(ls_train).to(device).long().cuda()
        
        data_grads = []
        for model in models:
            class_output,domain_output, rvs_domain_output = model(input_data=x_eval, alpha=0)
            pred = class_output.data.max(1, keepdim=True)[1]
            if pred.item() != y_eval.item():
                continue
            loss_func = wLoss().cuda()
            loss = loss_func(class_output, y_eval, ls_train_batch)
            model.zero_grad()
            loss.backward()
            data_grad = x_eval.grad.data
            data_grads.append(data_grad)
        return data_grads


    def generate(self,save_path,alpha):
        filename = f"{save_path}.pickle"
        if os.path.exists(filename):
            return
        dataset = self.__load_dataset()
        x_data, y_data, s_data = dataset["x_data"],dataset["y_data"],dataset["s_data"]
        
        feat_mu = np.mean(x_data,axis=0)
        feat_st = np.std(x_data, axis=0)
        
        x_data  = normalization_ops(feat_mu, feat_st, x_data)
        ls_trains = np.eye(4)[y_data]
        #TODO it is sample
        models = self.__load_model(0,1)

        new_dataset = {"x_data":[],"y_data":[],"s_data":[]}
        for x,y,s,ls_train in tqdm(zip(x_data,y_data,s_data,ls_trains),total=len(x_data)):
            x = np.array(x)
            gradients = self.get_model_result(models,x,y,ls_train)
            fsgm_data = self.fsgm(x,gradients,alpha)
            for fsgm_data_unit in fsgm_data:
                new_dataset["x_data"].append(fsgm_data_unit[0])
                new_dataset["y_data"].append(y)
                new_dataset["s_data"].append(s)
        
        makedirs(save_path)
        with open(filename, 'wb') as handle:
            pickle.dump(new_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

            