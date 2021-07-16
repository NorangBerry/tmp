# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:29:56 2020
@ref: https://github.com/fungtion/DANN_py3/blob/master/main.py
@author: Youngdo Ahn
"""
import torch
import torch.nn as nn
from functions import ReverseLayerF

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self._build()
        for p in self.parameters():
            p.requires_grad = True

    def _build(self):
        self.feature = nn.Sequential()
        self.feature.add_module('f_fc1', nn.Linear(1582, 1024))
        #self.feature.add_module('f_bn1', nn.BatchNorm1d(1024))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_drop1', nn.Dropout())
        self.feature.add_module('f_fc2', nn.Linear(1024, 1024))
        #self.feature.add_module('f_bn2', nn.BatchNorm1d(1024))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_drop2', nn.Dropout())
        #self.feature.add_module('f_fc3', nn.Linear(1024, 1024))
        #self.feature.add_module('f_bn2', nn.BatchNorm1d(1024))
        #self.feature.add_module('f_relu3', nn.PReLU())#(True))
        #self.feature.add_module('f_drop3', nn.Dropout())
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(1024, 512))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(512, 512))
        self.class_classifier.add_module('c_drop2', nn.Dropout())
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        #self.class_classifier.add_module('c_fc3', nn.Linear(512, 512))
        #self.class_classifier.add_module('c_drop3', nn.Dropout())
        #self.class_classifier.add_module('c_relu3', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(512, 4))
        #self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(1024, 512))
        self.domain_classifier.add_module('d_drop1', nn.Dropout())
        self.domain_classifier.add_module('d_relu1', nn.PReLU())#(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(512, 512))
        self.domain_classifier.add_module('d_drop2', nn.Dropout())
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(512, 1))#domains)) 
        self.domain_classifier.add_module('d_sigmoid', nn.Sigmoid())#(dim=1))
        #self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

    def forward(self, input_data, alpha):
        #input_data = input_data.expand(input_data.data.shape[0], 1582)
        input_data = input_data.view(input_data.data.shape[0], 1582)
        feature = self.feature(input_data)
        feature = feature.view(input_data.data.shape[0],1024) #feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(feature)#.view(-1)
        rvs_domain_output = self.domain_classifier(reverse_feature).view(-1)

        return class_output, domain_output, rvs_domain_output