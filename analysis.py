# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:39:06 2021

@author: Youngdo Ahn
"""
from const import ROOT_PATH
import os
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from functions import normalization_ops, makedirs, load_unlabeled_corpus, load_emotion_corpus, wc_confusion, wc_evaluation, load_soft_labels
from model import BaseModel
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
#import torch.backends.cudnn as tbc
#tbc.deterministic = True
#tbc.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set by cases 'CREMA-D','IEMOCAP','MSPIMPROV' 'EMO-DB', 'ETRI18','DEMoS'
train_dataset_list = np.sort(['CREMA-D','MSPIMPROV', 'MSPPodcast']) # 'IEMOCAP','MSPIMPROV', 'CREMA-D', 'IEMOCAP', 'MSPPodcast', 'CREMA-D',
AUX_dataset_list = ['Librispeechn'] # 'none', 'Voxceleb2', 'none', Librispeech 'none','Librispeech', ,'KsponSpeech28','KsponSpeech'
test_dataset_list  = ['IEMOCAP'] #KorSE18  'ETRI18' 'EMO-DB' 'MMKOR' 'MSPPodcast', 'EMO-DB', 'KorSE18','ETRI18',
AUX_Mode_list      = ['LHae1'] # ,'none' 'none',,'dann' 'none','dann' , 'spae', 'spVae'
# 'none','fc','V', 'fcV', 'VS'
# 'none','fc','V','fcV','VS','fcVS','sp','spV'
# 'ae', 'fcae', 'Vae', 'fcVae', 'VSae', 'fcVSae', 'spae', 'spVae'
AUX_DB_modes = ['NEU', 'ENT', 'ae','aeNEU','aeENT','dann', 'fcdann','dannNEU', 'dannENT','dannae','dannaeENT','dannaeNEU',
                            'Ndann', 'NdannENT','NdannNEU','aedd','aeddENT','aeddNEU','fcaedd', 'fcVaedd',
                            'er','erNEU','erENT','aestd','dannstd', 'fcae','fcVae', 'Cae','VSae','CSae','Cdann','VSdann','CSdann',
                            'fcVSae','fcVSdann']
pAUX_DB_modes = ['NEU', 'ENT', 'aeNEU', 'aeENT', 'dannNEU', 'dannENT','dannaeENT','dannaeNEU', 'NdannENT','NdannNEU','aeddENT','aeddNEU','erNEU','erENT'] 
RUN_Option = ['soft_mse'] #  source_confusion AUX_DB_pred T-SNE soft_mse opSNE
Model_NAME = 'DG0612' # Basemodel Base_DG51
lr = 2e-4
batch_size = 32
n_seeds = 5
n_epochs = 100
n_patience = 10 
DATA_PATH = "E:/FEW_DBs/"
FEAT_NAME = "emobase2010"
alpha = 0.1
gap = 0.05#0.1


ap1 = 0.0
ap2 = 0.0
ap3 = 0.0 #4.0


'''
ap1_list = [0.0]            # Label smoothing value    [0.0, 0.1, 0.2, 0.3] 
                                           # FC                       [0.5, 1.0, 2.0]
                                           # Domain Difficulty value  [1., 2., 3., 4., 5.]
                                           # LHAA Ambiguity Augmentation   [0.01, 0.05, 0.1]
ap2_list = [4.0]            # Confident penalty value  [5.0, 4.0, 3.0, 2.0, 1.0, 0.5]
                                           # Confident difficulty penaly [1.0, 2.0, 3.0]
ap3_list = [4.0]        # Domain weight            [1.0, 2.0, 3.0, 4.0, 5.0]
                                           # Aux_module ('ae'..)      [0.1, 0.5, 1.0]
'''

ap1_AUX = ['Vae','fcae','fcVae','fcaedd','fcVaedd','aestd','dannstd','dann', 'fcdann', 'fcVdann', 'Vdann', 'Vaedd',
           'ae','aeENT','aeNEU', 'aedd','aeddENT','aeddNEU','er','erENT','erNEU',
           'Cae','VSae','CSae','Cdann','VSdann','CSdann','fcVSae','fcVSdann','spae','spdann', 'spVae','spVSae']
ap2_AUX = ['fc','fcae','fcV','fcVae','nonestd','aestd','dannstd','fcVaedd','fcaedd', 'fcVdann',
           'fcVS','fcVSae','fcVSdann',
           'NEU', 'ENT', 'dannNEU', 'dannENT', 'aeENT','aeNEU', 'aeddENT','aeddNEU','er','erENT','erNEU','fcdann']
ap3_AUX = ['er','erENT','erNEU', 'V', 'VS', 'fcV', 'fcVS', 'Vae', 'VSae', 'fcV', 'fcVS', 'fcVae','fcVSae'] 

ap1_list = [0.0] # adv. or rec. loss [1., 0.5, 0.1]
ap2_list = [0.0] #[0.5, 1., 2.] # focal loss!! 0.5 2.
ap3_list = [0.0] #[0.1, 0.3, 0.5, 0.7, 0.9] # soft-label panalty


x_train = []
y_train = []
x_valid = []
y_valid = []
tr_n_samples = 100000
for D_tmp in train_dataset_list:
    data_path = os.path.join(DATA_PATH+D_tmp, FEAT_NAME)
    x_tr_tmp, y_tr_tmp, x_vl_tmp, y_vl_tmp = load_emotion_corpus(D_tmp, data_path)
    x_train.append(x_tr_tmp)
    y_train.append(y_tr_tmp)
    x_valid.append(x_vl_tmp)
    y_valid.append(y_vl_tmp)
    if tr_n_samples > len(y_tr_tmp):
        tr_n_samples = len(y_tr_tmp)
n_minibatch = int(np.floor(tr_n_samples/batch_size))
x_tmp = np.concatenate(x_train)
feat_mu = np.mean(x_tmp,axis=0)
feat_st = np.std(x_tmp, axis=0)

path_dbs = ''
n_domain = len(train_dataset_list)
for dn, D_tmp in enumerate (train_dataset_list):
    x_train[dn]  = normalization_ops(feat_mu, feat_st, x_train[dn])
    x_valid[dn]  = normalization_ops(feat_mu, feat_st, x_valid[dn])
    path_dbs += D_tmp

del x_tmp

AUX_DB = 'Librispeech' #AUX_dataset_list[0]
data_path = os.path.join(ROOT_PATH+AUX_DB, FEAT_NAME)
x_aux  = load_unlabeled_corpus(data_path, AUX_DB)
x_aux  = normalization_ops(feat_mu, feat_st, x_aux) 
x_aux  = torch.Tensor(x_aux).to(device).cuda()
AUX_DB = 'Librispeechn'
### Confusion matrix
mode_list = []
cnf_mean_list  = []
cnf_std_list  = []
dcnf_mean_list  = []
dcnf_std_list  = []
dcnf_case_list = []
if 'source_confusion' in RUN_Option:
    print("Valid Corpus: %s" %(train_dataset_list))
    for AUX_DB in AUX_dataset_list:
        for AUX_Mode in AUX_Mode_list:
            train_path = os.path.join(ROOT_PATH+path_dbs, FEAT_NAME)
            if AUX_Mode in AUX_DB_modes and AUX_DB != 'none':
            #if AUX_Mode in ['NEU', 'ENT', 'ae','aeNEU','aeENT','dann', 'dannNEU', 'dannENT','dannae','dannaeENT','dannaeNEU','Ndann', 'NdannENT','NdannNEU'] and AUX_DB != 'none':
                train_path = os.path.join(ROOT_PATH+path_dbs+AUX_DB+'a', FEAT_NAME)
            elif (AUX_DB != 'none') or (AUX_Mode != 'none' and len(train_dataset_list)==1):
                continue
            elif AUX_DB == 'none' and AUX_Mode in pAUX_DB_modes:
            #elif AUX_DB == 'none' and AUX_Mode in ['NEU', 'ENT', 'aeNEU', 'aeENT', 'dannNEU', 'dannENT','dannaeENT','dannaeNEU', 'NdannENT','NdannNEU']:
                continue
            ap3 = 0.
            if 'ae' in AUX_Mode or 'dann' in AUX_Mode :
                ap1 = 1.
            else:
                ap1 = 0.
            if 'fc' in AUX_Mode:
                ap2 = 1.
            else:
                ap2 = 0.
            mode_list.append(AUX_Mode+'_'+AUX_DB)
            tmp_cnf_list = []    
            for ti, tc in enumerate(train_dataset_list):
                dtmp_cnf_list = []  
                for seed in range(n_seeds):
                    my_net = torch.load('%s/%s/%s_ap1%s_ap2%s_ap3%s_seed%s.pth' %(train_path, Model_NAME, AUX_Mode,
                                                        str(ap1), str(ap2), str(ap3), str(seed)))
                    #my_net = torch.load('%s/%s/%s_seed%s.pth' %(train_path, Model_NAME, AUX_Mode, str(seed)))
                    tmp_cnf_list.append(wc_confusion(my_net, x_valid[ti], y_valid[ti], 0, device))
                    dtmp_cnf_list.append(wc_confusion(my_net, x_valid[ti], y_valid[ti], 0, device))
                dcnf_mean_list.append(np.mean(dtmp_cnf_list,axis=0).round(decimals=2))
                dcnf_std_list.append(np.std(dtmp_cnf_list,axis=0))
                dcnf_case_list.append('val_'+tc+'_mod_'+ AUX_Mode)
            cnf_mean_list.append(np.mean(tmp_cnf_list,axis=0).round(decimals=2))
            cnf_std_list.append(np.std(tmp_cnf_list,axis=0))
elif 'target_confusion' in RUN_Option:
    for cc in test_dataset_list:
        cc_path = os.path.join(DATA_PATH+cc, FEAT_NAME)
        x_target, y_target = load_unlabeled_corpus(cc_path, cc)
        x_target  = normalization_ops(feat_mu, feat_st, x_target)
        print("Cross Corpus: %s" %(cc))
        for AUX_Mode in AUX_Mode_list:
            ap3 = 0.
            if 'ae' in AUX_Mode or 'dann' in AUX_Mode :
                ap1 = 1.
            else:
                ap1 = 0.
            if 'fc' in AUX_Mode:
                ap2 = 1.
            else:
                ap2 = 0.
            for AUX_DB in AUX_dataset_list:
                train_path = os.path.join(ROOT_PATH+path_dbs, FEAT_NAME)
                if AUX_Mode in AUX_DB_modes and AUX_DB != 'none':
                #if AUX_Mode in ['NEU', 'ENT', 'ae','aeNEU','aeENT','dann', 'dannNEU', 'dannENT','dannae','dannaeENT','dannaeNEU','Ndann', 'NdannENT','NdannNEU'] and AUX_DB != 'none':
                    train_path = os.path.join(ROOT_PATH+path_dbs+AUX_DB+'a', FEAT_NAME)
                elif (AUX_DB != 'none') or (AUX_Mode != 'none' and len(train_dataset_list)==1):
                    continue
                elif AUX_DB == 'none' and AUX_Mode in pAUX_DB_modes:
                #elif AUX_DB == 'none' and AUX_Mode in ['NEU', 'ENT', 'aeNEU', 'aeENT', 'dannNEU', 'dannENT','dannaeENT','dannaeNEU', 'NdannENT','NdannNEU']:
                    continue
                mode_list.append(cc+'_'+AUX_Mode+'_'+AUX_DB)
                tmp_cnf_list = []
                wa_list = []
                ua_list = []
                for seed in range(n_seeds):
                    my_net = torch.load('%s/%s/%s_ap1%s_ap2%s_ap3%s_seed%s.pth' %(train_path, Model_NAME, AUX_Mode,
                                    str(ap1), str(ap2), str(ap3), str(seed)))
                    #my_net = torch.load('%s/%s/%s_b%s_g%s_seed%s.pth' %(train_path, Model_NAME, AUX_Mode, str(beta), str(gamma), str(seed)))
                    #my_net = torch.load('%s/%s/%s_seed%s.pth' %(train_path, Model_NAME, AUX_Mode, str(seed)))
                    tmp_cnf_list.append(wc_confusion(my_net, x_target, y_target, 0, device))
                    tmp_wa, tmp_ua = wc_evaluation(my_net, [x_target], [y_target], 0, device)
                    wa_list.append(tmp_wa[0])
                    ua_list.append(tmp_ua[0])
                #print("## AUX: %s+%s ##: WA %.2f-%.3f *** UA %.2f-%.3f *** bb%.2f, gg%.2f" %(AUX_Mode, AUX_DB,
                #            np.mean(wa_list), np.std(wa_list), np.mean(ua_list), np.std(ua_list), beta, gamma))
                cnf_mean_list.append(np.mean(tmp_cnf_list,axis=0).round(decimals=2))
                cnf_std_list.append(np.std(tmp_cnf_list,axis=0))
### Class Probability Librispeech

color_list = ['green', 'yellow', 'blue', 'red']
if 'AUX_DB_pred' in RUN_Option:
    print("Valid Corpus: %s" %(train_dataset_list))
    pred_list = []
      
    for AUX_DB in AUX_dataset_list:
        for AUX_Mode in AUX_Mode_list:
            train_path = os.path.join(ROOT_PATH+path_dbs, FEAT_NAME)
            if AUX_Mode in AUX_DB_modes and AUX_DB != 'none':
                train_path = os.path.join(ROOT_PATH+path_dbs+AUX_DB+'a', FEAT_NAME)
            elif (AUX_DB != 'none') or (AUX_Mode != 'none' and len(train_dataset_list)==1):
                continue
            elif AUX_DB == 'none' and AUX_Mode in pAUX_DB_modes:
                continue
            
            for ap1 in ap1_list:
                if not AUX_Mode in ap1_AUX:
                    ap1 = 0.0
                for ap2 in ap2_list:
                    if not AUX_Mode in ap2_AUX:
                        ap2 = 0.0
                    if AUX_Mode in ['fcV','fcVae','fcVaedd','fcVdann'] and ap2 == 0:
                        continue
                    for ap3 in ap3_list:
                        if not AUX_Mode in ap3_AUX:
                            ap3 = 0.0
                            
                        mode_list.append(AUX_Mode+'_'+AUX_DB)
                        tmp_pred_list = []
                        for seed in range(n_seeds): # n_seeds
                            try:
                                my_net = torch.load('%s/%s/%s_ap1%s_ap2%s_ap3%s_seed%s.pth' %(train_path, Model_NAME, AUX_Mode,
                                                                            str(ap1), str(ap2), str(ap3), str(seed)))
                            except:
                                print("Skipped model: %s/%s/%s_ap1%s_ap2%s_ap3%s_seed%s.pth" %(train_path, Model_NAME, AUX_Mode,
                                                                            str(ap1), str(ap2), str(ap3), str(seed)))
                                continue
                            my_net.eval()
                            class_output, _, _ = my_net(input_data=x_aux, alpha=alpha)
                            tmp_pred_list.append(np.array(F.softmax(class_output,1).tolist()))
                        pred_list.append(np.concatenate(tmp_pred_list,axis=0))
                        if not AUX_Mode in ap3_AUX:
                            break
                    if not AUX_Mode in ap2_AUX:
                        break
                if not AUX_Mode in ap1_AUX:
                    break       
    for mi, mn in enumerate(mode_list):
        for ci in range(4):
            g_list = []
            x_list = []
            for gi in range(int(1/gap)):
                x_list.append((gi+0.5)*gap)
                g_list.append(sum(np.logical_and(gi*gap<pred_list[mi].T[ci],pred_list[mi].T[ci]<=(gi+1)*gap)))
            plt.plot(x_list, g_list, color=color_list[ci], marker='o', linestyle='--')
            #plt.hist(pred_list[mi].T[ci], bins=1000, density=True, alpha=0.5, histtype='step')
        plt.legend(['neu','hap','sad','ang'])
        plt.xlabel('softmax output')
        plt.ylabel('#samples')
        plt.title('Method: %s, Target: %s' %(mn, AUX_dataset_list[-1]))
        plt.show()
elif 'opSNE' in RUN_Option:
    h_clr_emo = ['green', 'orange', 'blue', 'red']
    h_clr_dom = ['brown', 'gray', 'purple', 'black']
    h_grp_emo = ['neu', 'hap', 'sad', 'ang']
    d_tmp = []
    for dn, Dom in enumerate(train_dataset_list):
        d_tmp += list(np.ones(len(y_train[dn])).astype(int)*int(dn))
    d_tmp = np.array(d_tmp)
    y_tmp = np.concatenate(y_train)
    print("Corpus Visualization: %s" %(train_dataset_list))
    tsne = TSNE(learning_rate=300, n_iter=1000)
    f_tmp = np.concatenate(x_train)
    transformed = tsne.fit_transform(f_tmp)
    #fig_fd_path = '%s/%s/fig' %(train_path, Model_NAME)
    #fig_path = '%s/%s_op.pdf' %(fig_fd_path, AUX_Mode)
    plt.figure(figsize=(18,8))
    plt.subplot(121)
    for dn, Dom in enumerate(train_dataset_list):
        hx, hy = transformed[d_tmp==dn].T
        plt.scatter(hx,hy,alpha=0.8, c=h_clr_dom[dn], edgecolors='none', s = 30, label=Dom)
    plt.legend()
    plt.title('Domain representation')
    plt.subplot(122)
    for ci in range(4):
        hx, hy = transformed[y_tmp==ci].T
        plt.scatter(hx,hy,alpha=0.8, c=h_clr_emo[ci], edgecolors='none', s = 30, label=h_grp_emo[ci])       
    plt.legend()
    #plt.savefig(fig_path,dpi=400)
    #plt.close()   
    plt.title('Class representation')
    plt.show()  
        
elif 'T-SNE' in RUN_Option:
    h_clr_emo = ['green', 'orange', 'blue', 'red']
    h_clr_dom = ['brown', 'gray', 'purple', 'black']
    h_grp_emo = ['neu', 'hap', 'sad', 'ang']
    ignore_emo = []
    d_tmp = [] # 0:neu, 1:hap, 2:sad, 3:ang
    for dn, Dom in enumerate(train_dataset_list):
        d_tmp += list(np.ones(len(y_valid[dn])).astype(int)*int(dn))
    for cc in test_dataset_list:
        cc_path = os.path.join(ROOT_PATH+cc, FEAT_NAME)
        x_target, y_target = load_unlabeled_corpus(cc_path, cc)
        x_target  = normalization_ops(feat_mu, feat_st, x_target)
        y_tmp = np.concatenate(y_valid)
        y_tmp = np.array(list(y_tmp)+list(y_target))
        d_tmp += list(np.ones(len(y_target)).astype(int)*int(dn+1))
        d_tmp = np.array(d_tmp)
        print("Valid Corpus: %s Target Corpus: %s" %(train_dataset_list, test_dataset_list))
        tsne = TSNE(learning_rate=300, n_iter=1000)
        for AUX_DB in AUX_dataset_list:
            for AUX_Mode in AUX_Mode_list:
                train_path = os.path.join(ROOT_PATH+path_dbs, FEAT_NAME)
                if AUX_Mode in AUX_DB_modes and AUX_DB != 'none':
                    train_path = os.path.join(ROOT_PATH+path_dbs+AUX_DB+'a', FEAT_NAME)
                elif (AUX_DB != 'none') or (AUX_Mode != 'none' and len(train_dataset_list)==1):
                    continue
                elif AUX_DB == 'none' and AUX_Mode in pAUX_DB_modes:
                    continue
                
                for ap1 in ap1_list:
                    if not AUX_Mode in ap1_AUX:
                        ap1 = 0.0
                    for ap2 in ap2_list:
                        if not AUX_Mode in ap2_AUX:
                            ap2 = 0.0
                        if AUX_Mode in ['fcV','fcVae','fcVaedd','fcVdann'] and ap2 == 0:
                            continue
                        for ap3 in ap3_list:
                            if not AUX_Mode in ap3_AUX:
                                ap3 = 0.0
                            mode_list.append(AUX_Mode+'_'+AUX_DB)
                            tmp_pred_list = []    
                            print(mode_list[-1])
                            #for seed in range(n_seeds):
                                #my_net = torch.load('%s/%s/%s_seed%s.pth' %(train_path, Model_NAME, AUX_Mode, str(seed)))
                            my_net =torch.load('%s/%s/%s_ap1%s_ap2%s_ap3%s_best.pth' %(train_path, Model_NAME, AUX_Mode, str(ap1), str(ap2), str(ap3)))
                            my_net.eval()
                            f_list = []
                            for dn, Dom in enumerate(train_dataset_list):
                                f_list.append(np.array(my_net.feature(torch.Tensor(x_valid[dn]).to(device).cuda()).tolist()))
                            f_list.append(np.array(my_net.feature(torch.Tensor(x_target).to(device).cuda()).tolist()))
                            f_tmp = np.concatenate(f_list)
                            #for iemo in ignore_emo:
                            #    f_tmp = f_tmp[y_tmp!=iemo]
                            #    y_tmp = y_tmp[y_tmp!=iemo]
            
                            transformed = tsne.fit_transform(f_tmp)
                            fig_fd_path = '%s/%s/fig' %(train_path, Model_NAME)
                            #makedirs(fig_fd_path)
                            #fig_path = '%s/%s_seed%s.pdf' %(fig_fd_path, AUX_Mode, str(seed))
                            fig_path = '%s/%s_best.pdf' %(fig_fd_path, AUX_Mode)
                            plt.figure(figsize=(18,8))
                            plt.subplot(121)
                            for dn, Dom in enumerate(train_dataset_list):
                                hx, hy = transformed[d_tmp==dn].T
                                plt.scatter(hx,hy,alpha=0.8, c=h_clr_dom[dn], edgecolors='none', s = 30, label=Dom)
                            hx, hy = transformed[d_tmp==dn+1].T
                            plt.scatter(hx,hy,alpha=0.8, c=h_clr_dom[dn+1], edgecolors='none', s = 30, label=cc)
                            plt.legend()
                            plt.subplot(122)
                            for ci in range(4):
                                hx, hy = transformed[y_tmp==ci].T
                                plt.scatter(hx,hy,alpha=0.8, c=h_clr_emo[ci], edgecolors='none', s = 30, label=h_grp_emo[ci])       
                            plt.legend()
                            #plt.savefig(fig_path,dpi=400)
                            #plt.close()   
                            plt.title('Target: %s, Method: %s, ASR: %s' %(cc, AUX_Mode, AUX_DB))
                            plt.show()   
                            if not AUX_Mode in ap3_AUX:
                                break
                        if not AUX_Mode in ap2_AUX:
                            break
                    if not AUX_Mode in ap1_AUX:
                        break       
elif 'T-SNE_sad_ang' in RUN_Option:
    h_clr_emo = ['green', 'orange', 'blue', 'red']
    h_clr_dom = ['brown', 'gray', 'purple', 'black']
    h_grp_emo = ['neu', 'hap', 'sad', 'ang']
    emo_skip  = [0,1,1,0]
    ignore_emo = []
    d_tmp = [] # 0:neu, 1:hap, 2:sad, 3:ang
    for dn, Dom in enumerate(train_dataset_list):
        d_tmp += list(np.ones(len(y_valid[dn])).astype(int)*int(dn))
    for cc in test_dataset_list:
        cc_path = os.path.join(ROOT_PATH+cc, FEAT_NAME)
        x_target, y_target = load_unlabeled_corpus(cc_path, cc)
        x_target  = normalization_ops(feat_mu, feat_st, x_target)
        y_tmp = np.concatenate(y_valid)
        y_tmp = np.array(list(y_tmp)+list(y_target))
        y_skip = []
        for yt in y_tmp:
            if emo_skip[yt] == 0:
                y_skip.append(0)
            else:
                y_skip.append(1)
        y_skip = np.array(y_skip)
        d_tmp += list(np.ones(len(y_target)).astype(int)*int(dn+1))
        d_tmp = np.array(d_tmp)
        print("Valid Corpus: %s Target Corpus: %s" %(train_dataset_list, test_dataset_list))
        tsne = TSNE(learning_rate=300, n_iter=1000)
        for AUX_DB in AUX_dataset_list:
            for AUX_Mode in AUX_Mode_list:
                train_path = os.path.join(ROOT_PATH+path_dbs, FEAT_NAME)
                if AUX_Mode in ['NEU', 'ENT', 'ae','aeNEU','aeENT','dann', 'dannNEU', 'dannENT','dannae','dannaeENT','dannaeNEU','Ndann', 'NdannENT','NdannNEU'] and AUX_DB != 'none':
                    train_path = os.path.join(ROOT_PATH+path_dbs+AUX_DB+'a', FEAT_NAME)
                elif (AUX_DB != 'none') or (AUX_Mode != 'none' and len(train_dataset_list)==1):
                    continue
                elif AUX_DB == 'none' and AUX_Mode in ['NEU', 'ENT', 'aeNEU', 'aeENT', 'dannNEU', 'dannENT','dannaeENT','dannaeNEU', 'NdannENT','NdannNEU']:
                    continue
                mode_list.append(AUX_Mode+'_'+AUX_DB)
                tmp_pred_list = []    
                print(mode_list[-1])
                #for seed in range(n_seeds):
                    #my_net = torch.load('%s/%s/%s_seed%s.pth' %(train_path, Model_NAME, AUX_Mode, str(seed)))
                my_net = torch.load('%s/%s/%s_best.pth' %(train_path, Model_NAME, AUX_Mode))
                my_net.eval()
                f_list = []
                for dn, Dom in enumerate(train_dataset_list):
                    f_list.append(np.array(my_net.feature(torch.Tensor(x_valid[dn]).to(device).cuda()).tolist()))
                f_list.append(np.array(my_net.feature(torch.Tensor(x_target).to(device).cuda()).tolist()))
                f_tmp = np.concatenate(f_list)
                #for iemo in ignore_emo:
                #    f_tmp = f_tmp[y_tmp!=iemo]
                #    y_tmp = y_tmp[y_tmp!=iemo]

                transformed = tsne.fit_transform(f_tmp)
                fig_fd_path = '%s/%s/fig' %(train_path, Model_NAME)
                makedirs(fig_fd_path)
                #fig_path = '%s/%s_seed%s.pdf' %(fig_fd_path, AUX_Mode, str(seed))
                plt.figure(figsize=(18,8))
                #plt.title('Taget: %s, Method: %s, ASR: %s' %(cc, AUX_Mode, AUX_DB))
                plt.subplot(121)
                for dn, Dom in enumerate(train_dataset_list):
                    hx, hy = transformed[np.logical_and(d_tmp==dn,y_skip)].T
                    plt.scatter(hx,hy,alpha=0.8, c=h_clr_dom[dn], edgecolors='none', s = 30, label=Dom)
                hx, hy = transformed[d_tmp==dn+1].T
                plt.scatter(hx,hy,alpha=0.8, c=h_clr_dom[dn+1], edgecolors='none', s = 30, label=cc)
                plt.legend()
                plt.subplot(122)
                emo_name = ''
                for ci in range(4):
                    if emo_skip[ci] == 1:
                        emo_name = emo_name + '_' + h_grp_emo[ci]
                        #emo_name.append('_'+h_grp_emo[ci])
                        hx, hy = transformed[y_tmp==(ci)].T
                        plt.scatter(hx,hy,alpha=0.8, c=h_clr_emo[ci], edgecolors='none', s = 30, label=h_grp_emo[ci])       
                plt.legend()
                
                fig_path = '%s/%s_best%s.pdf' %(fig_fd_path, AUX_Mode,emo_name)
                plt.savefig(fig_path,dpi=400)
                plt.close()  
elif 'bt_gm' in RUN_Option:
    tgt_dom_mod = []
    tgt_dom_avg = []
    beta_list = [0.2, 0.4, 0.6, 0.8, 1.0]
    gamma_list = [0.2, 0.4, 0.6, 0.8, 1.0]
elif 'soft_mse' in RUN_Option:
    cc = test_dataset_list[0]
    print("Target corpus: %s" %(cc))
    cc_path = os.path.join(DATA_PATH+cc, FEAT_NAME)
    x_target, yh_target, ys_target = load_soft_labels(cc_path, cc)
    x_target  = normalization_ops(feat_mu, feat_st, x_target)
    yo_target = np.eye(4)[yh_target]
    #load model

    tgt_dom_mod = []
    tgt_mse_avg = []
    tgt_mse1_avg = []
    tgt_ua_avg = []
    for AUX_Mode in AUX_Mode_list:
        wa_list = []
        ua_list = []
        mse_list = []
        mse_list1 = []
        cos_list = []
        cos_list1 = []
        for seed in range(n_seeds):
            #train_path = os.path.join(ROOT_PATH+path_dbs, FEAT_NAME)
            train_path = os.path.join(ROOT_PATH+path_dbs+AUX_DB+'a', FEAT_NAME)
            #my_net = torch.load('%s/%s/%s_best.pth' %(train_path, Model_NAME, AUX_Mode))
            #my_net = torch.load('%s/%s/%s_seed%s.pth' %(train_path, Model_NAME, AUX_Mode, str(seed)))
            my_net = torch.load('%s/%s/%s_ap1%s_ap2%s_ap3%s_seed%s.pth' %(train_path, Model_NAME, AUX_Mode,
                                                        str(ap1), str(ap2), str(ap3), str(seed)))
            my_net.eval()
            tmp_wa, tmp_ua = wc_evaluation(my_net, [x_target], [yh_target], 0, device)
            x_eval = torch.Tensor(x_target).to(device).cuda()
            class_output, _, _ = my_net(x_eval, alpha=0)
            wa_list.append(tmp_wa[0])
            ua_list.append(tmp_ua[0])
            class_output = F.softmax(class_output,1)
            mse_list.append(((np.array(class_output.tolist())-ys_target)**2).mean())
            mse_list1.append(((yo_target-ys_target)**2).mean())
            cos_list.append(cosine_similarity(np.array(class_output.tolist()),ys_target).diagonal().mean())
            cos_list1.append(cosine_similarity(yo_target,ys_target).diagonal().mean())

        print("## AUX: %s+%s ##: WA %.2f-%.3f *** UA %.2f-%.3f *** MSE %.4f *** MSE1 %.4f *** COS %.4f *** COS1 %.4f ***ap1%.2f_ap2%.2f_ap3%.2f" %(AUX_Mode, AUX_DB,
                                    np.mean(wa_list), np.std(wa_list), np.mean(ua_list), np.std(ua_list),
                                    np.sqrt(np.mean(mse_list)),np.sqrt(np.mean(mse_list1)), 
                                    np.mean(cos_list),np.mean(cos_list1),ap1, ap2, ap3))
        tgt_dom_mod.append('tg_%s_md_%s_db_%s_ap1%.2f_ap2%.2f_ap3%.2f' %(cc, AUX_Mode, AUX_DB, ap1, ap2, ap3))
        tgt_mse_avg.append(np.mean(mse_list).round(4))
        tgt_mse1_avg.append(np.mean(mse_list1).round(4))
        tgt_ua_avg.append(np.mean(ua_list).round(2))
    
'''
filename = "%s.pickle" %('test_results4')
with open(filename, 'rb') as handle:
    data1 = pickle.load(handle)
        
'''
    