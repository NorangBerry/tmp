# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:47:00 2021

@author: Youngdo Ahn
"""


from const import ROOT_PATH
import random
import os 
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.nn import functional as F
import numpy as np
import torch
import random
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from functions import normalization_ops, wc_evaluation, wLoss, makedirs
from model import BaseModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_IEMOCAP_WC(train_path, fold, c_name, phase='train'):
    train_filename = "%s.pickle"  %(train_path)
    
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
    x_data  = np.array(data['x_data'],dtype=float)
    y_data = np.array(data['yh_data'],dtype=int)
    ys_data = np.array(data['y_data'],dtype=int)
    s_data  = np.array(data['s_data'],dtype=int)
    data_list = y_data < 4     
    x_data  = x_data[data_list]
    y_data  = y_data[data_list]
    ys_data = ys_data[data_list]
    s_data  = s_data[data_list]
    
    test_list  = s_data==fold
    if fold % 2 == 0:
        valid_list = s_data==fold+1
    else:
        valid_list = s_data==fold-1
    #valid_list  = np.logical_or(test_list,valid_list)
    x_valid     = x_data[valid_list]
    y_valid     = y_data[valid_list] 
    x_test     = x_data[test_list]
    y_test     = y_data[test_list] 
    
    train_list = np.logical_not(np.logical_or(test_list,valid_list))
    x_train    = x_data[train_list]
    y_train    = y_data[train_list]

    
    ys_data = ys_data[test_list]
    ys_data = ys_data.T[:4]
    ys_data = (ys_data/ys_data.sum(0)).T
    del data 
    return x_train, y_train, x_valid, y_valid, x_test, y_test, ys_data
        
def load_CREMAD_WC(train_path, corpus):
    train_filename = "%s.pickle"  %(train_path)
    
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
    ys_test = data['ys_data'][te_list==1]
    #ys_test = np.eye(4)[y_test]
    ys_test = ys_test.T[:4]
    ys_test = (ys_test/ys_test.sum(0)).T
    del data 
    return x_train, y_train, x_valid, y_valid, x_test, y_test, ys_test
    #'''
    
def load_DEMoS_WC(train_path, corpus):
    train_filename = "%s.pickle"  %(train_path)
    
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
    tot_spk = 69
    tra_spk = 8*tot_spk//10 # 48
    val_spk = 1*tot_spk//10 # 6
    
    rnd_spk = random.sample(range(1,tot_spk+1),tot_spk)
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
    x_train = data['x_data'][tr_list==1].astype('float') # about 3300 
    y_train = data['y_data'][tr_list==1]
    
    x_valid = data['x_data'][vl_list==1].astype('float') # about 360
    y_valid = data['y_data'][vl_list==1]
    
    x_test = data['x_data'][te_list==1].astype('float') # about 360
    y_test = data['y_data'][te_list==1]

    del data 
    return x_train, y_train, x_valid, y_valid, x_test, y_test    
        
def load_emotion_corpus_WC(corpus, train_path, fold):
    y_test_soft = None
    if corpus in ['IEMOCAP', 'MSPIMPROV']:
        if corpus== 'IEMOCAP':
            foldt=fold%10
        else:
            foldt = fold
        x_train, y_train, x_valid, y_valid, x_test, y_test, y_test_soft = load_IEMOCAP_WC(train_path, foldt, corpus)
    elif corpus in ['CREMA-D']:
        x_train, y_train, x_valid, y_valid, x_test, y_test, y_test_soft = load_CREMAD_WC(train_path, corpus)
    elif corpus in ['DEMoS']:
        x_train, y_train, x_valid, y_valid, x_test, y_test = load_DEMoS_WC(train_path, corpus)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, y_test_soft

# DB Loading problem!
RUN_Option = 'WC'
n_seeds = 5#5
n_patience = 5#5#5
batch_size = 32
lr = 2e-4
n_epochs = 100
DATA_PATH = "E:/FEW_DBs/"
FEAT_NAME = "emobase2010"
Model_NAME = 'WC0716_JY'
#train_dataset_list = ['DEMoS']#['IEMOCAP']
# 'CREMA-D','IEMOCAP','MSP4IMPROV','MSPPodcast'
DATASET_LIST = list(np.sort(['CREMA-D'])) # ['DEMoS']# ['CREMA-D','IEMOCAP','MSPIMPROV']
DO_mode = 'train' # test train
# WA_train = []
# WA_valid = []
# WA_test  = []
# UA_train = []
UA_valid = []
UA_test  = []
# EUC_train = []
# EUC_valid = []
EUC_test  = []
# COS_train = []
# COS_valid = []
COS_test  = []
for DATASET in DATASET_LIST:
    train_dataset_list = [DATASET] # DATASET_LIST # 
    train_path = os.path.join(os.path.join(ROOT_PATH,DATASET), FEAT_NAME)
    if 'WC' in RUN_Option:
        n_fold = 10
        if 'MSPIMPROV' == DATASET:
            n_fold = 12#12
        elif 'CREMA-D' == DATASET:
            n_fold = 1
        fold_UA_valid = []
        fold_UA_test = []
        # fold_EUC_valid = []
        fold_EUC_test = []
        fold_COS_valid = []
        fold_COS_test = []
        for fold in range(n_fold):
            print('******** Dataset Loading ***********')
            print('***SRC %s  FOLD %d***********' %(train_dataset_list, fold))
            x_train = []
            y_train = []            
            x_valid = []
            y_valid = []
            x_test  = []
            y_test  = []
            ys_test  = []
            ls_train = []
            tr_n_samples = 100000
            for D_tmp in train_dataset_list:
                data_path = os.path.join(DATA_PATH+D_tmp, FEAT_NAME)
                x_tr_tmp, y_tr_tmp, x_vl_tmp, y_vl_tmp, x_te_tmp, y_te_tmp, ys_te_tmp = load_emotion_corpus_WC(D_tmp, data_path,fold)
                x_train.append(x_tr_tmp)
                y_train.append(y_tr_tmp)
                x_valid.append(x_vl_tmp)
                y_valid.append(y_vl_tmp)
                x_test.append(x_te_tmp)
                y_test.append(y_te_tmp)
                ys_test.append(ys_te_tmp)
                if tr_n_samples > len(y_tr_tmp):
                    tr_n_samples = len(y_tr_tmp)
            for dn, yl in enumerate (y_train):
                ls_train.append(np.eye(4)[yl])
            n_minibatch = int(np.floor(tr_n_samples/batch_size))
            x_tmp = np.concatenate(x_train)
            feat_mu = np.mean(x_tmp,axis=0)
            feat_st = np.std(x_tmp, axis=0)
            
            path_dbs = ''
            n_domain = len(train_dataset_list)
            for dn, D_tmp in enumerate (train_dataset_list):
                x_train[dn]  = normalization_ops(feat_mu, feat_st, x_train[dn])
                x_valid[dn]  = normalization_ops(feat_mu, feat_st, x_valid[dn])
                x_test[dn]   = normalization_ops(feat_mu, feat_st, x_test[dn])
                path_dbs += D_tmp

            for seed in range(n_seeds):
                best_UA_valid = [0.]*len(train_dataset_list)
                best_UA_test  = [0.]*len(train_dataset_list)
                best_EUC_test  = [0.]*len(train_dataset_list)
                best_COS_test  = [0.]*len(train_dataset_list)
                if DO_mode == 'train':
                    print('SEED %d start' %(seed))
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    my_net = BaseModel(domains=n_domain)
    
                    for p in my_net.parameters():
                        p.requires_grad = True
    
                    # setup optimizer
                    optimizer = optim.Adam(my_net.parameters(), lr=lr)
                    my_loss   = wLoss().cuda() #FocalLoss().cuda() #
                    #my_loss.gamma     = 0. # Simple CE loss
                    my_net    = my_net.cuda()
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=2) 
                    best_score = -1000
                    best_UA  = 0.
                    pt_times = 0
    
                    for epoch in range(n_epochs):
                        # Start an epoch (training)
                        rd_arr = []
                        m_train= []
                        for dn, Dom in enumerate(train_dataset_list):
                            rd_arr.append(random.sample(range(len(y_train[dn])),tr_n_samples))
                        my_net.train()
                        
                        for bc in range(n_minibatch):
                            optimizer.zero_grad()
                            L_tot = 0.
                            x_train_batch = []
                            y_train_batch = []
                            ls_train_batch = []
                            d_train_batch = []
                            for dn in range(n_domain): 
                                x_train_batch.append(torch.Tensor(x_train[dn][rd_arr[dn][bc*batch_size:batch_size*(bc+1)]]))
                                y_train_batch.append(torch.Tensor(y_train[dn][rd_arr[dn][bc*batch_size:batch_size*(bc+1)]]))
                                ls_train_batch.append(torch.Tensor(ls_train[dn][rd_arr[dn][bc*batch_size:batch_size*(bc+1)]]))
                                d_train_batch.append(torch.Tensor(np.ones(batch_size,)*dn))
                            x_train_batch = torch.cat(x_train_batch,dim=0).to(device).cuda()
                            y_train_batch = torch.cat(y_train_batch,dim=0).to(device).long().cuda()
                            ls_train_batch = torch.cat(ls_train_batch,dim=0).to(device).long().cuda()
                            d_train_batch = torch.cat(d_train_batch,dim=0).to(device).long().cuda()
                            
                            class_output, _, _ = my_net(input_data=x_train_batch, alpha=0)
                            class_logit_list = []
                            label_list = []
                            ls_label_list = []
                            weight_list = []
                            for i in range(len(train_dataset_list)):
                                class_logit_list.append(class_output[d_train_batch==i])
                                label_list.append(y_train_batch[d_train_batch==i])
                                ls_label_list.append(ls_train_batch[d_train_batch==i])
                                weight = torch.zeros(4).to(device)
                                for j in range(4):
                                    weight[j] = 0 if (label_list[-1]==j).sum() == 0 else 1.0 / (label_list[-1]==j).sum().float() 
                                    #weight[j] = 0 if weight[j] == 1 else weight[j]
                                weight = weight / (weight.sum() + 1e-8)
                                weight_list.append(weight)
                            L_total = 0.
                            for p, l, w, ls in zip(class_logit_list, label_list, weight_list, ls_label_list):
                                if p is None:
                                    continue
                                my_loss.alpha     =  w #None
                                L_total += my_loss(p, l, ls) /float(len(train_dataset_list))   
                            L_total.backward()
                            optimizer.step()
                            del x_train_batch, y_train_batch, d_train_batch 
                        # Start an epoch (validation)
                        my_net.eval()
                        tmp_wa_list = []
                        tmp_ua_list = []
                        for dn, Dom in enumerate(train_dataset_list):
                            tmp_wa, tmp_ua = wc_evaluation(my_net, 
                                            [x_train[dn], x_valid[dn]], [y_train[dn], y_valid[dn]], 0, device)
                            tmp_wa_list.append(tmp_wa)
                            tmp_ua_list.append(tmp_ua)
                        tmp_wa = np.mean(tmp_wa_list,0)
                        tmp_ua = np.mean(tmp_ua_list,0)
                        tmp_score = tmp_ua[1] 
                        
                        print("[Tra] wa: %.2f ua: %.2f [Val] wa: %.2f ua: %.2f" % (tmp_wa[0],tmp_ua[0], tmp_wa[1],tmp_ua[1]))
                        scheduler.step(tmp_score)
                        if tmp_score > best_score:
                            best_score = tmp_score
                            best_UA = tmp_ua[1]
    
                            for dn, Dom in enumerate(train_dataset_list):
                                _, tmp_ua = wc_evaluation(my_net, [x_valid[dn], x_test[dn]], \
                                                          [y_valid[dn],y_test[dn]], 0, device)
                                best_UA_valid[dn] = tmp_ua[0]
                                best_UA_test[dn] = tmp_ua[-1]
                            print("new_acc!")
                            makedirs('%s/%s' %(train_path, Model_NAME))
                            torch.save(my_net, '%s/%s/WC_fold%s_seed%s.pth' %(train_path, Model_NAME, str(fold), str(seed)))
                            pt_times = 0
                        else:
                            pt_times += 1
                            if pt_times == n_patience:
                                break
                elif DO_mode == 'test':
                    #load model and out!
                    my_net = torch.load('%s/%s/WC_fold%s_seed%s.pth' %(train_path, Model_NAME, str(fold), str(seed))) 
                    my_net.eval()
                    
                    for dn, Dom in enumerate(train_dataset_list):
                        _, tmp_ua = wc_evaluation(my_net, [x_valid[dn], x_test[dn]], \
                                                              [y_valid[dn],y_test[dn]], 0, device)
                        best_UA_valid[dn] = tmp_ua[0]
                        best_UA_test[dn] = tmp_ua[-1]
                    
                        x_eval = torch.Tensor(x_test[dn]).to(device).cuda()
                        class_output, _, _ = my_net(x_eval, alpha=0)
                        class_output = F.softmax(class_output,1)
                        best_EUC_test[dn] = np.sqrt(((np.array(class_output.tolist())-ys_test[dn])**2).sum(axis=-1)).mean()
                        best_COS_test[dn] = cosine_similarity(np.array(class_output.tolist()),ys_test[dn]).diagonal().mean()
                    fold_EUC_test.append(best_EUC_test)
                    fold_COS_test.append(best_COS_test)
        
                # seed end
                fold_UA_valid.append(best_UA_valid)
                fold_UA_test.append(best_UA_test)
            # fold end (do nothing)
        # DB end
        UA_valid.append(fold_UA_valid)
        UA_test.append(fold_UA_test)
        EUC_test.append(fold_EUC_test)
        COS_test.append(fold_COS_test)
        print("WC Domain [%s] valid UA: %.2f-%.4f test UA %.2f-%.4f" %(Dom, np.mean(UA_valid[-1]),np.std(UA_valid[-1]),
                                                            np.mean(UA_test[-1]),np.std(UA_test[-1])))
'''
for dn, Dom in enumerate(DATASET_LIST):
    print("WC Domain [%s] valid UA: %.2f - %.4f test UA %.2f - %.4f" %(Dom, np.mean(UA_valid[dn]),np.std(UA_valid[dn]),\
                                                          np.mean(UA_test[dn]),np.std(UA_test[dn])))
## Single
for dn, Dom in enumerate(DATASET_LIST):
    print("WC Domain [%s] test EUC: %.4f-%.4f COS %.4f-%.4f" %(Dom, np.mean(EUC_test[dn]),np.std(EUC_test[dn]),\
                                                               np.mean(COS_test[dn]),np.std(COS_test[dn])))
## Multi
for dn, Dom in enumerate(DATASET_LIST):
    print("WC Domain [%s] test EUC: %.4f-%.4f COS %.4f-%.4f" %(Dom, np.mean(EUC_test[0],axis=0)[dn],np.std(EUC_test[0],axis=0)[dn],\
                                                          np.mean(COS_test[0],axis=0)[dn],np.std(COS_test[0],axis=0)[dn]))
'''
for dn, Dom in enumerate(DATASET_LIST):
    UAt_valid = []
    UAt_test  = []
    for tn in range(len(fold_UA_valid)):
        #UA_valid.append(fold_UA_valid[tn*(len(DATASET_LIST))+dn])
        UAt_valid.append(fold_UA_valid[tn][dn])
        UAt_test.append(fold_UA_test[tn][dn])
    print("WC Domain [%s] valid UA: %.2f - %.4f test UA %.2f - %.4f" %(Dom, \
                np.mean(UAt_valid),np.std(UAt_valid),np.mean(UAt_test),np.std(UAt_test)))
        
    #print("WC Domain [%s] valid UA: %.2f - %.4f test UA %.2f - %.4f" %(Dom, np.mean(UA_valid[dn]),np.std(UA_valid[dn]),\
    #                                                        np.mean(UA_test[dn]),np.std(UA_test[dn])))
                