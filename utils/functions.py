# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:18:19 2020

@author: Youngdo Ahn
"""
import math
from torch.autograd import Function
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch
import os

def rms(wav:np.ndarray):
    return np.sqrt(np.mean(wav**2))

def snr(signal:np.ndarray,noise:np.ndarray):
    return 20 * math.log(rms(signal)/rms(noise),10)
    

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
class FocalLoss(nn.Module):
    #https://github.com/clcarwin/focal_loss_pytorch
    def __init__(self, gamma=0, alpha=None, size_average=True, relu=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.relu  = relu
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input,1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        # argp = input.argmax(axis=1).view(-1,1)
        # lgc = (argp==target).long()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.relu:
            argp = input.argmax(axis=1).view(-1,1)
            argp = (argp!=target).long().view(-1)
            #loss *= argp
            loss = loss[argp==1]
        if self.size_average: return loss.mean()
        else: return loss.sum()
class wLoss(nn.Module):
    #https://github.com/clcarwin/focal_loss_pytorch
    def __init__(self, gamma=0, alpha=None, size_average=True, loss_name='CE'):
        super(wLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.loss_name = loss_name

    def forward(self, input, target, mtarget):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        #'''
        if self.loss_name == 'MS': 
            pt = F.softmax(input, 1)
            loss = ((pt - Variable(mtarget))**2).mean(-1)
        elif self.loss_name == 'CE':
            #pt = torch.sigmoid(input)
            #loss = ((pt - mtarget)**2).mean(-1)
            logpt = F.log_softmax(input,1) 
            loss = -(mtarget*logpt).sum(-1) 
        else:
            raise ValueError("loss_name CHECK")

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            loss = loss * Variable(at)
        if self.size_average: return loss.mean()
        else: return loss.sum()

def normalization_ops(feat_mu, feat_st, x_data):
    x_data = np.nan_to_num((x_data - feat_mu) / feat_st) #np.clip(, -10, 10)
    # x_data = np.clip(x_data, -10, 10)
    cut_list = np.abs(x_data)>10
    x_data[cut_list] = 0
    return x_data

def makedirs(path):
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)

def wc_evaluation(model, x_list, y_list, alpha, device):
    model.eval()
    eval_wa = []
    eval_ua = []
    for xn, x_eval in enumerate(x_list):
        x_eval = torch.Tensor(x_eval).to(device).cuda()
        y_eval = torch.Tensor(y_list[xn]).to(device).long().cuda()
        class_output, _, _ = model(input_data=x_eval, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        eval_wa.append(accuracy_score(y_eval.data.cpu(),pred.data.cpu())*100)
        eval_ua.append(balanced_accuracy_score(y_eval.data.cpu(),pred.data.cpu())*100)
        del x_eval, y_eval
    return eval_wa, eval_ua