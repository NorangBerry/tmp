from train.base import DataType, ModelRunner
from utils.setting import ROOT_PATH,device, get_model_dir
from utils.data_loader import load_emotion_corpus_WC
import random
import os 
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import numpy as np
import torch
import random
from utils.functions import normalization_ops, wc_evaluation, makedirs


class Trainer(ModelRunner):
    def __init__(self,dataset):
        self.init_setting()
        self.dataset = dataset
        self.model_dir = get_model_dir(self.dataset)
        self.data_path = os.path.join(ROOT_PATH,self.dataset,"opensmile","emobase2010")

    def set_data(self,fold):
        
        x_train, y_train, x_valid, y_valid, x_test, y_test, ys_test = load_emotion_corpus_WC(self.dataset, self.data_path,fold)
        tr_n_samples = min(100000,len(y_train))

        ls_train = np.eye(4)[y_train]
        n_minibatch = int(np.floor(tr_n_samples/self.setting.batch_size))

        feat_mu = np.mean(x_train,axis=0)
        feat_st = np.std(x_train, axis=0)
        
        x_train  = normalization_ops(feat_mu, feat_st, x_train)
        x_valid  = normalization_ops(feat_mu, feat_st, x_valid)
        x_test   = normalization_ops(feat_mu, feat_st, x_test)

        self.dataloader = {
            DataType.X_TRAIN:x_train,
            DataType.Y_TRAIN:y_train,
            DataType.X_VALIDATION:x_valid,
            DataType.Y_VALIDATION:y_valid,
            DataType.X_TEST:x_test,
            DataType.Y_TEST:y_test,
            DataType.YS_TEST:ys_test,
        }
        return ls_train,n_minibatch,tr_n_samples

    def train_epoch(self,ls_train,tr_n_samples,n_minibatch):
        # Start an epoch (training)
        random_samples = random.sample(range(len(self.dataloader[DataType.Y_TRAIN])),tr_n_samples)
        self.model.train()
        for bc in range(n_minibatch):
            self.optimizer.zero_grad()

            samples = random_samples[bc*self.setting.batch_size : (bc+1)*self.setting.batch_size]
            x_train_batch = torch.Tensor(self.dataloader[DataType.X_TRAIN][samples]).to(device).cuda()
            y_train_batch = torch.Tensor(self.dataloader[DataType.Y_TRAIN][samples]).to(device).long().cuda()
            ls_train_batch = torch.Tensor(ls_train[samples]).to(device).long().cuda()
            
            class_output, _, _ = self.model(input_data=x_train_batch, alpha=0)

            weight = torch.zeros(4).to(device)
            for j in range(4):
                weight[j] = 0 if (y_train_batch==j).sum() == 0 else 1.0 / (y_train_batch==j).sum().float() 
            weight = weight / (weight.sum() + 1e-8)

            self.my_loss.alpha =  weight
            L_total = self.my_loss(class_output, y_train_batch, ls_train_batch) /1.0
            L_total.backward()
            self.optimizer.step()

            del x_train_batch, y_train_batch 
        # Start an epoch (validation)
        self.model.eval()
        tmp_wa, tmp_ua = wc_evaluation(self.model, 
                        [self.dataloader[DataType.X_TRAIN], self.dataloader[DataType.X_VALIDATION]], 
                        [self.dataloader[DataType.Y_TRAIN], self.dataloader[DataType.Y_VALIDATION]], 0, device)
        tmp_score = tmp_ua[1] 
        
        print("[Tra] wa: %.2f ua: %.2f [Val] wa: %.2f ua: %.2f" % (tmp_wa[0],tmp_ua[0], tmp_wa[1],tmp_ua[1]))
        self.scheduler.step(tmp_score)
        return tmp_score, tmp_ua
    
    def run_seed(self,seed,ls_train,tr_n_samples,n_minibatch,fold):
        self.set_random_seed(seed)
        self.init_network()
        
        pt_times = 0
        best_score = -1000
        best_UA_valid = 0.
        best_UA_test  = 0.
        
        for epoch in range(self.setting.n_epochs):
            tmp_score, tmp_ua = self.train_epoch(ls_train,tr_n_samples,n_minibatch)
            if tmp_score > best_score:
                best_score = tmp_score
                _, tmp_ua = wc_evaluation(self.model, [self.dataloader[DataType.X_VALIDATION], self.dataloader[DataType.X_TEST]],
                                            [self.dataloader[DataType.Y_VALIDATION],self.dataloader[DataType.Y_TEST]], 0, device)
                best_UA_valid = tmp_ua[0]
                best_UA_test = tmp_ua[-1]
                print("new_acc!")
                pt_times = 0
            else:
                pt_times += 1
                if pt_times == self.setting.n_patience:
                    break
                
        makedirs(self.model_dir)
        torch.save(self.model, os.path.join(self.model_dir,f"WC_fold{fold}_seed{seed}.pth"))

        return best_UA_valid, best_UA_test

    def run_fold(self,fold):
        print('******** Dataset Loading ***********')
        print(f'***SRC {self.dataset}  FOLD {fold}***********')
        super().run_fold(fold)