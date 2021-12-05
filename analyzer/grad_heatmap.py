import os
import pickle
from matplotlib import pyplot as plt

import torch

from utils.functions import normalization_ops, wLoss
import numpy as np

from utils.setting import get_model_dir, get_pickle_path, device

class GradientHeatmap:
    def __init__(self):
        data_path = get_pickle_path("CREMA-D")
        x_data, y_data,ls_trains = self.load_pickle_data(data_path)
        self.dataset = { "x_data":x_data, "y_data":y_data, "ls_trains":ls_trains }
        self.model = self.load_model()

    def load_pickle_data(self,data_path):
        train_filename = f"{data_path}.pickle"
        with open(train_filename, 'rb') as handle:
            data = pickle.load(handle)
            x_data = data["x_data"]
            y_data = data["y_data"]

            feat_mu = np.mean(x_data,axis=0)
            feat_st = np.std(x_data, axis=0)
            
            x_data  = normalization_ops(feat_mu, feat_st, x_data)

            ls_trains = np.eye(4)[y_data]
            return x_data, y_data, ls_trains

    def load_model(self):
        model_path = os.path.join(get_model_dir("CREMA-D"),f"WC_fold{0}_seed{0}.pth")
        my_net:torch.nn.Module = torch.load(model_path)
        my_net.eval()
        return my_net

    def process(self):
        for x_eval,y_eval,ls_train in zip(self.dataset["x_data"],self.dataset["y_data"],self.dataset["ls_trains"]):
            gradient = self.run_model(x_eval,y_eval,ls_train)
            if gradient != None:
                self.save_gradient_fig(gradient)

            
    
    def run_model(self,x_eval,y_eval,ls_train):
        x_eval = torch.Tensor(x_eval).to(device).cuda().unsqueeze(0)
        y_eval = torch.Tensor([y_eval]).to(device).long().cuda()
        x_eval.requires_grad = True
        
        ls_train_batch = torch.Tensor(ls_train).to(device).long().cuda()
        class_output,domain_output, rvs_domain_output = self.model(input_data=x_eval, alpha=0)
        pred = class_output.data.max(1, keepdim=True)[1]
        if pred.item() != y_eval.item():
                return None
        loss_func = wLoss().cuda()
        loss = loss_func(class_output, y_eval, ls_train_batch)
        self.model.zero_grad()
        loss.backward()
        return x_eval.grad.data.cpu()

    def save_gradient_fig(self,gradient):
        plt.rcParams["figure.figsize"] = 5,2
        gradient = gradient[0][672:693]
        y = np.array(gradient.unsqueeze(0))
        frame1 = plt.imshow(y, cmap="plasma", aspect="auto")
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        plt.show()