import os
import pickle

import torch
from utils.functions import makedirs, wLoss
import numpy as np

from utils.setting import get_dataset_folder, get_model_dir, device

class FgsmPickleMaker:
    def __init__(self,dataset,type,value):
        self.dataset:str = dataset
        self.data_path:str = get_dataset_folder(dataset,type,value)
        self.model_dir:str = get_model_dir(dataset,type,value)


    def __load_dataset(self):
        train_filename = f"{self.data_path}.pickle"
    
        with open(train_filename, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def __load_model(self,fold,seed):
        my_net = torch.load(os.path.join(self.model_dir,f"WC_fold{fold}_seed{seed}.pth"))
        my_net.eval()
        return my_net
        
    def fsgm(self):
        pass

    def get_model_result(self,model,x_data,y_data):
        x_eval = torch.Tensor(x_data).to(device).cuda()
        x_eval.requires_grad = True
        y_eval = torch.Tensor(y_data).to(device).long().cuda()
        class_output,domain_output, rvs_domain_output = model(input_data=x_eval, alpha=0)
        pred = class_output.data.max(1, keepdim=True)[1]
        if pred.item() != y_eval.item():
            return None, y_eval
        
        loss_func = wLoss().cuda()
        loss = loss_func(class_output, y_eval)
        model.zero_grad()
        loss.backward()
        data_grad = x_eval.grad.data
        return data_grad, pred


    def generate(self,save_path,alpha):
        dataset = self.__load_dataset()
        x_data, y_data = dataset["x_data"],dataset["y_data"]
        new_x_data = []
        model = self.__load_model()
        for x,y in zip(x_data,y_data):
            x = np.array(x)
            gradient, label= model(x)
            print(label,y)
            print(label.item(),y.item())
            if y.item() != label.item():
                new_x_data.append(x)
                continue
            new_x_data.append(self.fsgm(x,gradient,alpha))
        
        makedirs(save_path)

        dataset["x_data"] = new_x_data
        filename = os.path.join(save_path,"emobase2010.pickle")
        with open(filename, 'wb') as handle:
           pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

            