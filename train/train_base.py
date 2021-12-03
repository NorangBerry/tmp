from datetime import datetime
import gc
from train.base import DataType, ModelRunner
from utils.setting import device, get_model_dir, get_pickle_path
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
        self.data_path = get_pickle_path(self.dataset)

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
        del self.dataloader
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
        model_path = os.path.join(self.model_dir,f"WC_fold{fold}_seed{seed}.pth")
        if os.path.exists(model_path):
            return 0,0
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
        model_path = os.path.join(self.model_dir,f"WC_fold{fold}_seed{self.setting.n_seeds-1}.pth")
        if os.path.exists(model_path):
            return [0],[0]
        return super().run_fold(fold)

    def run(self):
        model_path = os.path.join(self.model_dir,f"WC_fold{self.get_n_fold()-1}_seed{self.setting.n_seeds-1}.pth")
        if os.path.exists(model_path):
            return
        super().run()


class NoiseTrainer(Trainer):
    def __init__(self,dataset,dB):
        super().__init__(dataset)
        self.model_dir = get_model_dir(self.dataset,"noisy",dB)

    def set_data(self,fold):
        x_train, y_train, x_valid, y_valid, x_test, y_test, ys_test = load_emotion_corpus_WC(self.dataset, self.data_path,fold)
        x_train2, y_train2, x_valid2, y_valid2, x_test2, y_test2, ys_test2 = load_emotion_corpus_WC(self.dataset, self.noise_path,fold)
        x_train = np.concatenate((x_train,x_train2),axis=0)
        y_train = np.concatenate((y_train,y_train2),axis=0)
        x_valid = np.concatenate((x_valid,x_valid2),axis=0)
        y_valid = np.concatenate((y_valid,y_valid2),axis=0)
        x_test = np.concatenate((x_test,x_test2),axis=0)
        y_test = np.concatenate((y_test,y_test2),axis=0)
        ys_test = np.concatenate((ys_test,ys_test2),axis=0)

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



class FgsmTrainer(Trainer):
    def __init__(self,dataset,epsilon):
        super().__init__(dataset)
        self.model_dir = get_model_dir(self.dataset,"gradient",epsilon,True)

    def set_data(self,fold):
        self.dataloader = {}
        gc.collect()
        begin = datetime.now()
        x_train, y_train, x_valid, y_valid, x_test, y_test, ys_test = load_emotion_corpus_WC(self.dataset, self.data_path,fold)
        x_train2, y_train2, x_valid2, y_valid2, x_test2, y_test2, ys_test2 = load_emotion_corpus_WC(self.dataset, self.noise_path,fold)
        
        end = datetime.now()
        print(f"load time {end-begin}")
        begin = end

        multiplier = len(x_train2)//len(x_train) 
        if len(x_train2) % len(x_train) != 0:
            multiplier += 1

        x_train = np.tile(x_train,(multiplier,1))
        x_train = np.concatenate((x_train,x_train2),axis=0)
        

        y_train = np.tile(y_train,multiplier)
        y_train = np.concatenate((y_train,y_train2),axis=0)

        x_valid = np.tile(x_valid,(multiplier,1))
        x_valid = np.concatenate((x_valid,x_valid2),axis=0)

        y_valid = np.tile(y_valid,multiplier)
        y_valid = np.concatenate((y_valid,y_valid2),axis=0)

        x_test = np.tile(x_test,(multiplier,1))
        x_test = np.concatenate((x_test,x_test2),axis=0)

        y_test = np.tile(y_test,multiplier)
        y_test = np.concatenate((y_test,y_test2),axis=0)

        ys_test = np.tile(ys_test,multiplier)
        ys_test = np.concatenate((ys_test,ys_test2),axis=0)

        
        # end = datetime.now()
        # print(f"merge time {end-begin}")
        # begin = end

        tr_n_samples = min(100000,len(y_train))

        ls_train = np.eye(4)[y_train]
        n_minibatch = int(np.floor(tr_n_samples/self.setting.batch_size))

        feat_mu = np.mean(x_train,axis=0)
        feat_st = np.std(x_train, axis=0)
        
        x_train  = normalization_ops(feat_mu, feat_st, x_train)
        x_valid  = normalization_ops(feat_mu, feat_st, x_valid)
        x_test   = normalization_ops(feat_mu, feat_st, x_test)

        end = datetime.now()
        print(f"norm time {end-begin}")
        begin = end

        # if hasattr(self,"dataloader") and self.dataloader != None:
        #     keys = self.dataloader.keys()
        #     for key in keys:
        #         del self.dataloader[key]
        self.dataloader = {
            DataType.X_TRAIN:x_train,
            DataType.Y_TRAIN:y_train,
            DataType.X_VALIDATION:x_valid,
            DataType.Y_VALIDATION:y_valid,
            DataType.X_TEST:x_test,
            DataType.Y_TEST:y_test,
            DataType.YS_TEST:ys_test,
        }
        gc.collect()
        return ls_train,n_minibatch,tr_n_samples