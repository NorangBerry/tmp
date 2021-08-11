# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:18:19 2020

@author: Youngdo Ahn
"""
from torch.autograd import Function
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch
import random
import pickle
import os
from sklearn.metrics import confusion_matrix

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
        
def load_emotion_corpus(corpus, train_path):
    if corpus in ['IEMOCAP', 'MSPIMPROV', 'EMO-DB', 'KorSE18']:
        x_train, y_train, x_valid, y_valid = load_IEMOCAP820(train_path, 0, corpus)
    elif corpus in ['CREMA-D', 'ETRI18']:
        x_train, y_train, x_valid, y_valid = load_CREMAD(train_path, corpus)
    elif corpus == 'MSPPodcast':
        x_train, y_train, x_valid, y_valid = load_MSPPodcast(train_path)  
    elif corpus == 'MMKOR':
        x_train, y_train, x_valid, y_valid = load_MMKOR(train_path)  
    elif corpus == 'DEMoS':
        x_train, y_train, x_valid, y_valid = load_DEMoS(train_path, corpus)  
    return x_train, y_train, x_valid, y_valid

def load_MMKOR(train_path):
    train_filename = "%s.pickle"  %(train_path)
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
    x_data  = np.array(data['x_data'],dtype=float)
    y_data = np.array(data['y_data'],dtype=int)
    p_data = np.array(data['p_data'],dtype=int)
    
    x_train = x_data[p_data<=4400] # 1~5600
    y_train = y_data[p_data<=4400] # 
    x_valid = x_data[p_data>4400]
    y_valid = y_data[p_data>4400]
    
    del data 
    
    return x_train, y_train, x_valid, y_valid
def load_DEMoS(train_path, corpus):
    train_filename = "%s.pickle"  %(train_path)
    
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
    tot_spk = 69
    tra_spk = 8*tot_spk//10 # 48
    val_spk = tot_spk-tra_spk#2*tot_spk//10 # 6
    
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
    
    #x_test = data['x_data'][te_list==1].astype('float') # about 360
    #y_test = data['y_data'][te_list==1]

    del data 
    return x_train, y_train, x_valid, y_valid 
def load_MSPPodcast(train_path):
    train_filename = "%s.pickle"  %(train_path)
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
    x_data  = np.array(data['x_data'],dtype=float)
    y_data = np.array(data['y_data'],dtype=int)
    p_data = np.array(data['p_data'],dtype=str)
    
    x_train = x_data[p_data=='Tn']
    y_train = y_data[p_data=='Tn']
    x_valid = x_data[p_data=='Vn']
    y_valid = y_data[p_data=='Vn']
    
    del data 
    
    return x_train, y_train, x_valid, y_valid
'''
def load_soft_labels(train_path, c_name):
    train_filename = "%s.pickle"  %(train_path)
    
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
    x_data  = np.array(data['x_data'],dtype=float)
    if c_name in ['MSPPodcast']:
        y_data = np.array(data['ys_data'],dtype=int)
        yh_data = np.array(data['y_data'],dtype=int)
        p_data = np.array(data['p_data'],dtype=str)
        data_list = np.logical_or(p_data == 'T1',p_data == 'T2')
        x_data = x_data[data_list]
        y_data = y_data[data_list]
        yh_data = yh_data[data_list]
    elif c_name in ['IEMOCAP', 'MSPIMPROV']:
        y_data = np.array(data['y_data'],dtype=int)
        yh_data = np.array(data['yh_data'],dtype=int)
    else:
        raise Exception("Not defined dataset!")
    data_list = yh_data < 4     
    x_data = x_data[data_list]
    yh_data = yh_data[data_list]
    y_data = y_data[data_list]
    y_tmp = y_data.T[:4]
    y_soft = (y_tmp/y_tmp.sum(0)).T
    del data         
    return x_data, yh_data, y_soft
'''
def load_soft_labels(train_path, c_name):
    train_filename = "%s.pickle"  %(train_path)
    
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
    x_data  = np.array(data['x_data'],dtype=float)
    if c_name in ['MSPPodcast']:
        y_data = np.array(data['ys_data'],dtype=int)
        yh_data = np.array(data['y_data'],dtype=int)
        p_data = np.array(data['p_data'],dtype=str)
        x_data  = x_data[p_data=='Tn']
        y_data  = y_data[p_data=='Tn']
        yh_data = yh_data[p_data=='Tn']
    elif c_name in ['IEMOCAP', 'MSPIMPROV']:
        y_data = np.array(data['y_data'],dtype=int)
        yh_data = np.array(data['yh_data'],dtype=int)
        s_data  = np.array(data['s_data'],dtype=int)
        train_list = np.logical_and(s_data!=0,s_data!=1)
        x_data = x_data[train_list]
        y_data = y_data[train_list]
        yh_data = yh_data[train_list]
    elif c_name in ['CREMA-D']:
        y_data = np.array(data['ys_data'],dtype=int)
        yh_data = np.array(data['y_data'],dtype=int)
        s_data  = np.array(data['s_data'],dtype=int)
        train_spk_num = 58
        train_list = s_data<train_spk_num+1
        x_data = x_data[train_list]
        y_data = y_data[train_list]
        yh_data = yh_data[train_list]
    else:
        raise Exception("Not defined dataset!")
    data_list = yh_data < 4     
    x_data = x_data[data_list]
    yh_data = yh_data[data_list]
    y_data = y_data[data_list]
    y_tmp = y_data.T[:4]
    y_soft = (y_tmp/y_tmp.sum(0)).T
    del data         
    return x_data, yh_data, y_soft

def load_soft_labels_target(train_path, c_name):
    train_filename = "%s.pickle"  %(train_path)
    
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
    if c_name in ['MSPPodcast']:
        y_data = np.array(data['ys_data'],dtype=int)
        yh_data = np.array(data['y_data'],dtype=int)
        p_data = np.array(data['p_data'],dtype=str)
        target_list = np.logical_or(p_data == 'T1',p_data == 'T2')
        y_data  = y_data[target_list]
        yh_data = yh_data[target_list]
    elif c_name in ['IEMOCAP', 'MSPIMPROV']:
        y_data = np.array(data['y_data'],dtype=int)
        yh_data = np.array(data['yh_data'],dtype=int)
    elif c_name in ['CREMA-D']:
        y_data = np.array(data['ys_data'],dtype=int)
        yh_data = np.array(data['y_data'],dtype=int)
    else:
        raise Exception("Not defined dataset!")
    data_list = yh_data < 4     
    yh_data = yh_data[data_list]
    y_data = y_data[data_list]
    y_tmp = y_data.T[:4]
    y_soft = (y_tmp/y_tmp.sum(0)).T
    del data         
    return y_soft

def load_soft_labels_G(train_path, c_name):
    train_filename = "%s.pickle"  %(train_path)
    
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
    x_data  = np.array(data['x_data'],dtype=float)
    if c_name in ['MSPPodcast']:
        y_data = np.array(data['ys_data'],dtype=int)
        yh_data = np.array(data['y_data'],dtype=int)
    elif c_name in ['IEMOCAP', 'MSPIMPROV']:
        y_data = np.array(data['y_data'],dtype=int)
        yh_data = np.array(data['yh_data'],dtype=int)
    else:
        raise Exception("Not defined dataset!")
    data_list = yh_data < 4     
    x_data = x_data[data_list]
    yh_data = yh_data[data_list]
    y_data = y_data[data_list]
    y_tmp = y_data.T[:4]
    y_soft = (y_tmp/y_tmp.sum(0)).T
    del data         
    return x_data, yh_data, y_soft
def load_IEMOCAP820(train_path, fold, c_name):
    train_filename = "%s.pickle"  %(train_path)
    
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
    x_data  = np.array(data['x_data'],dtype=float)
    if c_name in ['EMO-DB', 'KorSE18']:
        y_data = np.array(data['y_data'],dtype=int)
    else:
        y_data = np.array(data['yh_data'],dtype=int)
    s_data  = np.array(data['s_data'],dtype=int)
    data_list = y_data < 4     
    x_data = x_data[data_list]
    y_data = y_data[data_list]
    s_data = s_data[data_list]
    
    test_list  = s_data==fold
    if fold % 2 == 0:
        valid_list = s_data==fold+1
    else:
        valid_list = s_data==fold-1
    if c_name == 'KorSE18':
        test_list = np.logical_and(s_data>10*fold,s_data<10*(fold+1))
        if fold % 2 == 0:
            valid_list = np.logical_and(s_data>10*(fold+1),s_data<10*(fold+2))
        else:
            valid_list = np.logical_and(s_data>10*(fold-1),s_data<10*(fold+0))
    valid_list = np.logical_or(test_list,valid_list)
    x_valid     = x_data[valid_list]
    y_valid     = y_data[valid_list] 
    
    train_list = np.logical_not(valid_list)
    x_train    = x_data[train_list]
    y_train    = y_data[train_list]

    del data 
        
    return x_train, y_train, x_valid, y_valid
        
def load_CREMAD(train_path, corpus):
    train_filename = "%s.pickle"  %(train_path)
    
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
    train_spk_num = 58
    if corpus == 'ETRI18':
        train_spk_num = 20
    x_train = data['x_data'][data['s_data']<train_spk_num+1] # here can be added other DBs
    y_train = data['y_data'][data['s_data']<train_spk_num+1]
    x_valid = data['x_data'][data['s_data']>train_spk_num]
    y_valid = data['y_data'][data['s_data']>train_spk_num]

    del data 
    return x_train, y_train, x_valid, y_valid
        

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

def Loss_evaluation(model, x_list, y_list, dn, device):
    model.eval()
    eval_wa = []
    eval_ua = []
    eval_L_ce = []
    eval_L_adv = []
    eval_L_rec = []
    for xn, x_eval in enumerate(x_list):
        x_eval = torch.Tensor(x_eval).to(device).cuda()
        y_eval = torch.Tensor(y_list[xn]).to(device).long().cuda()
        d_eval = torch.Tensor(np.ones(len(x_eval),)*dn).long().cuda()
        class_output, feature_g, rvs_src_output = model(x_eval,0)
        weight = torch.zeros(4).to(device)
        for j in range(4):
            weight[j] = 0 if (y_eval[-1]==j).sum() == 0 else 1.0 / (y_eval[-1]==j).sum().float() 
            #weight[j] = 0 if weight[j] == 1 else weight[j]
        weight = weight / (weight.sum() + 1e-6)
        eval_L_ce.append(F.cross_entropy(class_output, y_eval, weight).data.cpu())
        eval_L_adv.append(0.)#(F.cross_entropy(rvs_src_output, d_eval).data.cpu())
        #if feature_g.shape[-1] == x_eval.shape[-1]:
        #eval_L_rec.append(float(torch.mean((feature_g-x_eval)**2)))
        #else:
        eval_L_rec.append(0)
            
        pred = class_output.data.max(1, keepdim=True)[1]
        eval_wa.append(accuracy_score(y_eval.data.cpu(),pred.data.cpu())*100)
        eval_ua.append(balanced_accuracy_score(y_eval.data.cpu(),pred.data.cpu())*100)
        del x_eval, y_eval
    return eval_wa, eval_ua, eval_L_ce, eval_L_adv, eval_L_rec


'''
class_logit_list = []
label_list = []
weight_list = []

for i in range(len(train_dataset_list)):
    class_logit_list.append(class_output[d_train_batch==i])
    label_list.append(y_train_batch[d_train_batch==i])
    weight = torch.zeros(4).to(device)
    for j in range(4):
        weight[j] = 0 if (label_list[-1]==j).sum() == 0 else 1.0 / (label_list[-1]==j).sum().float() 
        #weight[j] = 0 if weight[j] == 1 else weight[j]
    weight = weight / (weight.sum() + 1e-6)
    weight_list.append(weight)
ce_list = []
for p, l, w in zip(class_logit_list, label_list, weight_list):
    if p is None:
        continue
    ce_list.append(F.cross_entropy(p, l, weight=w)) #/ len(train_dataset_list))
    #err += ce_list[-1]
L_ce_std, L_ce_mean = torch.std_mean(torch.Tensor(ce_list))
L_tt += L_ce_mean
if AUX_Mode in ['nonestd', 'aestd', 'dannstd']:
    L_tt += L_ce_std
if AUX_Mode in ['dann', 'dannNEU', 'dannENT','dannae','dannaeENT','dannaeNEU']:
    L_adv = loss_domain(rvs_src_output, d_train_batch)*ap1
    L_tt += L_adv
if AUX_Mode in ['ae','aeNEU','aeENT','dannae','dannaeENT','dannaeNEU']:
    L_rec = loss_class_AE(feature_g, x_train_batch)*ap1
    L_tt += L_rec

'''
def wc_confusion(model, x_list, y_list, alpha, device):
    model.eval()
    x_eval = torch.Tensor(x_list).to(device).cuda()
    y_eval = torch.Tensor(y_list).to(device).long().cuda()
    class_output, _, _ = model(input_data=x_eval, alpha=alpha)
    pred = class_output.data.max(1, keepdim=True)[1]
    cnf_matrix = confusion_matrix(y_eval.data.cpu(),pred.data.cpu())
    cnf_matrix = np.transpose(cnf_matrix)
    cnf_matrix = cnf_matrix*100 / cnf_matrix.astype(np.int).sum(axis=0)
    cnf_matrix = np.transpose(cnf_matrix).astype(float)
    del x_eval, y_eval
    return cnf_matrix

def fsl_sampling(m_train, x_few_set, n_way, n_shot, n_query, m_sample):
    x_tmp_train = []
    if m_train:
        rd_classes = random.sample(range(0,n_way),n_way)
    else:
        rd_classes = range(n_way)
    #rd_sen = random.sample(range(0,12),1)[0]
    for nw in range(n_way):                   
        if m_sample == 'rnd':
            rd_tmp = random.sample(range(len(x_few_set[rd_classes[nw]])), n_shot+n_query)
            x_tmp_train.append(x_few_set[rd_classes[nw]][rd_tmp])
            
    x_tmp_train = np.array(x_tmp_train)
    
    xs_train = x_tmp_train[:,n_query:,:] # W,S,I       
    xs_train = xs_train.reshape(n_way*n_shot,-1)
    xq_train = x_tmp_train[:,:n_query,:] # W,Q,I
    xq_train = xq_train.reshape(n_query*n_way,-1) # QW,I    
              
    return xs_train, xq_train

def wc_fsl_evaluation(model, x_few_set, x_test_list, y_test_list, alpha, 
                    n_way, n_shot, n_query, n_ensemble, m_sample, device):
    model.eval()
    y_pred_list = []
    for nx, x_test in enumerate(x_test_list):
        y_pred_list.append(np.zeros((x_test.shape[0],n_way)))       
 
        x_test_list[nx] = torch.Tensor(x_test_list[nx]).to(device).cuda()
        y_test_list[nx] = torch.Tensor(y_test_list[nx]).to(device).long().cuda()
    for ne in range(n_ensemble):
        xs_train, _ = fsl_sampling(False, x_few_set, n_way, n_shot, n_query, m_sample)
        xs_train = torch.Tensor(xs_train).to(device).cuda()

        for nx, x_test in enumerate(x_test_list):
            class_output, d_output, _ = model(xs=xs_train, xq=x_test, alpha=alpha)
            y_pred_list[nx] += np.array(class_output.data.tolist())
    wa_list = []
    ua_list = []
    for nx, x_test in enumerate(x_test_list):
        wa_list.append(accuracy_score(y_test_list[nx].data.cpu(), y_pred_list[nx].argmax(1))*100)
        ua_list.append(balanced_accuracy_score(y_test_list[nx].data.cpu(), y_pred_list[nx].argmax(1))*100)
    
    del xs_train, x_test_list, y_test_list
    return wa_list, ua_list

def load_unlabeled_corpus(cc_path, corpus):#(cc_path, corpus, mu, st, fold):
    cc_filename = "%s.pickle"  %(cc_path)
    if corpus == 'KsponSpeech28':
        cc_filename = "%s.pickle"  %(cc_path[:11]+'/KsponSpeech/emobase2010')
    with open(cc_filename, 'rb') as handle:
        data = pickle.load(handle)
    x_cc = np.array(data['x_data'],dtype=float)
    if corpus in ['IEMOCAP', 'MSPIMPROV', 'MSPPodcast', 'EMO-DB', 'KorSE18', 'ETRI18', 'MMKOR', 'CREMA-D', 'DEMoS']:
        y_cc = data['y_data']
    else:
        if corpus == 'KsponSpeech28':
            x_cc = x_cc[:28539]
        return x_cc
    
    if corpus == 'IEMOCAP':
        yh_data = np.array(data['yh_data'],dtype=int)
        data_list = yh_data < 4
        x_cc = x_cc[data_list]
        y_cc = yh_data[data_list]
    elif corpus == 'MSPIMPROV':
        y_cc = np.array(data['yh_data'],dtype=int)
    if corpus in ['MSPIMPROV', 'KorSE18']:
        data_list = y_cc < 4
        x_cc = x_cc[data_list]
        y_cc = y_cc[data_list]
    elif corpus == 'MSPPodcast':
        p_cc = np.array(data['p_data'],dtype=str)
        data_list = np.logical_or(p_cc == 'T1',p_cc == 'T2')
        x_cc = x_cc[data_list]
        y_cc = y_cc[data_list]
    del data
    return x_cc, y_cc 

#def load_few_corpus(x_target, y_target, sup_num, corpus):#(cc_path, corpus, mu, st, fold):
#    
#    return x_tr, y_tr, x_te, y_te 
def load_sup_unlabeled_corpus(cc_path, corpus):#(cc_path, corpus, mu, st, fold):
    cc_filename = "%s.pickle"  %(cc_path)
    if corpus == 'KsponSpeech28':
        cc_filename = "%s.pickle"  %(cc_path[:11]+'/KsponSpeech/emobase2010')
    with open(cc_filename, 'rb') as handle:
        data = pickle.load(handle)
    x_cc = np.array(data['x_data'],dtype=float)
    if corpus in ['IEMOCAP', 'MSPIMPROV', 'MSPPodcast', 'EMO-DB', 'KorSE18', 'ETRI18', 'MMKOR', 'CREMA-D', 'DEMoS']:
        y_cc = data['y_data']
    else:
        if corpus == 'KsponSpeech28':
            x_cc = x_cc[:28539]
        return x_cc
    
    if corpus == 'IEMOCAP':
        yh_data = np.array(data['yh_data'],dtype=int)
        data_list = yh_data < 4
        x_cc = x_cc[data_list]
        y_cc = yh_data[data_list]
    elif corpus == 'MSPIMPROV':
        y_cc = np.array(data['yh_data'],dtype=int)
    if corpus in ['MSPIMPROV', 'KorSE18']:
        data_list = y_cc < 4
        x_cc = x_cc[data_list]
        y_cc = y_cc[data_list]
    elif corpus == 'MSPPodcast':
        p_cc = np.array(data['p_data'],dtype=str)
        data_list = np.logical_or(p_cc == 'T1',p_cc == 'T2')
        x_cc = x_cc[data_list]
        y_cc = y_cc[data_list]
    del data
    return x_cc, y_cc 
def cc_fsl_evaluation(cc_path, corpus, mu, st, device, model, alpha, 
                      n_way, n_shot, n_query, n_ensemble, x_few_set, m_sample):
    cc_filename = "%s.pickle"  %(cc_path)
    with open(cc_filename, 'rb') as handle:
        data = pickle.load(handle)
    x_cc = np.array(data['x_data'],dtype=float)
    y_cc = data['y_data']
    
    if corpus == 'IEMOCAP':
        yh_data = np.array(data['yh_data'],dtype=int)
        data_list = yh_data < 4
        x_cc = x_cc[data_list]
        y_cc = yh_data[data_list]
    elif corpus == 'MSPIMPROV':
        y_cc = np.array(data['yh_data'],dtype=int)
    if corpus in ['MSPIMPROV', 'KorSE18']:
        data_list = y_cc < 4
        x_cc = x_cc[data_list]
        y_cc = y_cc[data_list]
    x_cc   = normalization_ops(mu, st, x_cc)
    x_cc = torch.Tensor(x_cc).to(device).cuda()
    y_cc = torch.Tensor(y_cc).to(device).long().cuda()
    
    wa_list, ua_list = wc_fsl_evaluation(model, x_few_set, [x_cc], [y_cc], alpha, n_way, 
                                         n_shot, n_query, n_ensemble, m_sample, device)
    del x_cc, y_cc
    return wa_list[0], ua_list[0] 

def _compute_cls_loss(device, model, feature, label, domain, mode="self"):
    #https://github.com/sshan-zhao/DG_via_ER/blob/2082d3d1c3703c153b85b0f6ec67632839887b59/train.py#L98
    if model is not None:
        class_logit_list = []
        label_list = []
        weight_list = []
        for i in range(int(max(domain)+1)):
            if mode == "self":
                class_logit_list.append(model[i](feature[domain==i]))
                label_list.append(label[domain==i])
            else:
                class_logit_list.append(model[i](feature[domain!=i]))
                label_list.append(label[domain!=i])
            weight = torch.zeros(4).to(device)
            for j in range(4):
                weight[j] = 0 if (label_list[-1]==j).sum() == 0 else 1.0 / (label_list[-1]==j).sum().float() 
            weight = weight / weight.sum()
            weight_list.append(weight)
        loss = 0
        for p, l, w in zip(class_logit_list, label_list, weight_list):
            if p is None:
                continue
            loss += F.cross_entropy(p, l, weight=w) / int(max(domain)+1)
    else:
        loss = torch.zeros(1, requires_grad=True).to(device)
    
    return loss