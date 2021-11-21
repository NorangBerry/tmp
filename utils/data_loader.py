# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:47:00 2021

@author: Youngdo Ahn
"""

import random
import numpy as np
import random
import pickle

def load_IEMOCAP_WC(train_path, fold):
    train_filename = f"{train_path}.pickle"
    
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
    x_data  = np.array(data['x_data'],dtype=float)
    y_data = np.array(data['y_data'],dtype=int)
    s_data  = np.array(data['s_data'],dtype=int)
    data_list = y_data < 4     
    x_data  = x_data[data_list]
    y_data  = y_data[data_list]
    ys_data = []
    s_data  = s_data[data_list]
    
    if fold % 2 == 0:
        valid_list = s_data==(fold+1)
        test_list  = s_data==(fold+2)
    else:
        valid_list = s_data==(fold+1)
        test_list  = s_data==(fold)
    #valid_list  = np.logical_or(test_list,valid_list)
    x_valid     = x_data[valid_list]
    y_valid     = y_data[valid_list] 
    x_test     = x_data[test_list]
    y_test     = y_data[test_list] 
    train_list = np.logical_not(np.logical_or(test_list,valid_list))
    x_train    = x_data[train_list]
    y_train    = y_data[train_list]

    
    del data 
    return x_train, y_train, x_valid, y_valid, x_test, y_test, ys_data
        
def load_CREMAD_WC(train_path):
    train_filename = f"{train_path}.pickle"
    
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
    tot_spk = 91
    tra_spk = 8*tot_spk//10 # 72
    val_spk = 1*tot_spk//10 # 9
    
    rnd_spk = random.sample(range(tot_spk),tot_spk)
    tr_list = np.zeros(len(data['s_data']))
    vl_list = np.zeros(len(data['s_data']))
    te_list = np.zeros(len(data['s_data']))
    for si, spk in enumerate (data['s_data']):
        for rnd in rnd_spk:
            if spk == rnd:
                if rnd < tra_spk:
                    tr_list[si] = 1
                elif rnd < tra_spk+val_spk:
                    vl_list[si] = 1
                else:
                    te_list[si] = 1
    x_train = data['x_data'][tr_list==1] # 3874
    y_train = data['y_data'][tr_list==1]
    
    x_valid = data['x_data'][vl_list==1] # 485
    y_valid = data['y_data'][vl_list==1]
    
    x_test = data['x_data'][te_list==1] # 540
    y_test = data['y_data'][te_list==1]
    ys_test = []#data['ys_data'][te_list==1]
    #ys_test = np.eye(4)[y_test]
    # ys_test = ys_test.T[:4]
    # ys_test = (ys_test/ys_test.sum(0)).T
    del data 
    return x_train, y_train, x_valid, y_valid, x_test, y_test, ys_test
           
def load_emotion_corpus_WC(corpus, train_path, fold):
    y_test_soft = None
    if corpus in ['IEMOCAP']:
        x_train, y_train, x_valid, y_valid, x_test, y_test, y_test_soft = load_IEMOCAP_WC(train_path, fold)
    elif corpus in ['CREMA-D']:
        x_train, y_train, x_valid, y_valid, x_test, y_test, y_test_soft = load_CREMAD_WC(train_path)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, y_test_soft
