from numpy.random import random_sample
from run_wc import ROOT_PATH, load_emotion_corpus_WC, train_epoch
import random
import os 
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import torch
import random
from sklearn.metrics.pairwise import cosine_similarity
from functions import normalization_ops, wc_evaluation, wLoss, makedirs
from model import BaseModel
from enum import Enum

class DataType(Enum):
    X_TRAIN = 1
    X_TEST = 2
    X_VALIDATION = 3
    Y_TRAIN = 4
    Y_TEST = 5
    Y_VALIDATION = 6
    YS_TEST = 7


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DB Loading problem!
RUN_Option = 'WC'
n_seeds = 5
n_patience = 5
batch_size = 32
lr = 2e-4
n_epochs = 100
# ROOT_PATH = "E:/FEW_DBs/" #"X:disk1/Young/datasets/FEW_DBs/"
DATA_PATH = ROOT_PATH
FEAT_NAME = "emobase2010"
Model_NAME = 'WC0716_JY'
#train_dataset_list = ['DEMoS']#['IEMOCAP']
# 'CREMA-D','IEMOCAP','MSPIMPROV','MSPPodcast'
DATASET_LIST = list(np.sort(['CREMA-D'])) # ['DEMoS']# ['CREMA-D','IEMOCAP','MSPIMPROV']
DO_mode = 'train' # test train

class Trainer():
    def __init__(self):
        pass

    def set_data(self,DATASET, data_path,fold):
        x_train, y_train, x_valid, y_valid, x_test, y_test, ys_test = load_emotion_corpus_WC(DATASET, data_path,fold)

        tr_n_samples = min(100000,len(y_train))

        ls_train = np.eye(4)[y_train]
        n_minibatch = int(np.floor(tr_n_samples/batch_size))

        feat_mu = np.mean(x_train,axis=0)
        feat_st = np.std(x_train, axis=0)
        
        x_train  = normalization_ops(feat_mu, feat_st, x_train)
        x_valid  = normalization_ops(feat_mu, feat_st, x_valid)
        x_test   = normalization_ops(feat_mu, feat_st, x_test)

        self.dataset = {
            DataType.X_TRAIN:x_train,
            DataType.Y_TRAIN:y_train,
            DataType.X_VALIDATION:x_valid,
            DataType.Y_VALIDATION:y_valid,
            DataType.X_TEST:x_test,
            DataType.Y_TEST:y_test,
            DataType.YS_TEST:ys_test,
        }
        return ls_train,n_minibatch,tr_n_samples

    def set_random_seed(self,seed):
        print('SEED %d start' %(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_n_fold(self,dataset):
        n_fold = 10
        if 'MSPIMPROV' == dataset:
            n_fold = 12
        elif 'CREMA-D' == dataset:
            n_fold = 1
        return n_fold

    def init_network(self):
        self.model = BaseModel().cuda()
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.my_loss   = wLoss().cuda() #FocalLoss().cuda() #
        #my_loss.gamma     = 0. # Simple CE loss
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='max', patience=2) 

    def train_epoch(self,ls_train,batch_size,tr_n_samples,n_minibatch):
        # Start an epoch (training)
        random_samples = random.sample(range(len(self.dataset[DataType.Y_TRAIN])),tr_n_samples)
        self.model.train()
        for bc in range(n_minibatch):
            self.optimizer.zero_grad()

            samples = random_samples[bc*batch_size:batch_size*(bc+1)]
            x_train_batch = torch.Tensor(self.dataset[DataType.X_TRAIN][samples]).to(device).cuda()
            y_train_batch = torch.Tensor(self.dataset[DataType.Y_TRAIN][samples]).to(device).long().cuda()
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
                        [self.dataset[DataType.X_TRAIN], self.dataset[DataType.X_VALIDATION]], 
                        [self.dataset[DataType.Y_TRAIN], self.dataset[DataType.Y_VALIDATION]], 0, device)
        
        tmp_score = tmp_ua[1] 
        
        print("[Tra] wa: %.2f ua: %.2f [Val] wa: %.2f ua: %.2f" % (tmp_wa[0],tmp_ua[0], tmp_wa[1],tmp_ua[1]))
        self.scheduler.step(tmp_score)
        return tmp_score, tmp_ua
    
    def train_seed(self,seed,ls_train,tr_n_samples,n_minibatch,train_path,fold):
        self.set_random_seed(seed)
        self.init_network()
        
        pt_times = 0
        best_score = -1000
        best_UA_valid = 0.
        best_UA_test  = 0.
        
        for epoch in range(n_epochs):
            tmp_score, tmp_ua = self.train_epoch(ls_train,batch_size,tr_n_samples,n_minibatch)
            if tmp_score > best_score:
                best_score = tmp_score
                _, tmp_ua = wc_evaluation(self.model, [self.dataset[DataType.X_VALIDATION], self.dataset[DataType.X_TEST]],
                                            [self.dataset[DataType.Y_VALIDATION],self.dataset[DataType.Y_TEST]], 0, device)
                best_UA_valid = tmp_ua[0]
                best_UA_test = tmp_ua[-1]
                print("new_acc!")
                makedirs('%s/%s' %(train_path, Model_NAME))
                torch.save(self.model, '%s/%s/WC_fold%s_seed%s.pth' %(train_path, Model_NAME, str(fold), str(seed)))
                pt_times = 0
            else:
                pt_times += 1
                if pt_times == n_patience:
                    break
        return best_UA_valid, best_UA_test

    def train_fold(self,DATASET,fold):
        train_path = os.path.join(os.path.join(ROOT_PATH,DATASET), FEAT_NAME)
        data_path = os.path.join(os.path.join(DATA_PATH,DATASET), FEAT_NAME)

        print('******** Dataset Loading ***********')
        print('***SRC %s  FOLD %d***********' %(DATASET, fold))

        ls_train,n_minibatch,tr_n_samples = self.set_data(DATASET, data_path,fold)

        fold_EUC_test = []
        fold_COS_test = []
        fold_UA_valid = []
        fold_UA_test = []
        for seed in range(n_seeds):
            best_UA_valid = 0.
            best_UA_test  = 0.
            if DO_mode == 'train':
                best_UA_valid, best_UA_test = self.train_seed(seed,ls_train,tr_n_samples,n_minibatch,train_path,fold)
            elif DO_mode == 'test':
                my_net = torch.load('%s/%s/WC_fold%s_seed%s.pth' %(train_path, Model_NAME, str(fold), str(seed))) 
                best_UA_valid,best_UA_test,best_EUC_test,best_COS_test = self.test(my_net)

                fold_EUC_test.append(best_EUC_test)
                fold_COS_test.append(best_COS_test)
    
            # seed end
            fold_UA_valid.append([best_UA_valid])
            fold_UA_test.append([best_UA_test])
        return fold_EUC_test, fold_COS_test, fold_UA_valid, fold_UA_test
        # fold end (do nothing)

    def train(self): 
        # self.list_manager = ScoreManager()
        UA_valid, UA_test, EUC_test, COS_test = ([] for _ in range(4))

        for DATASET in DATASET_LIST:
            if 'WC' in RUN_Option:
                n_fold = self.get_n_fold(DATASET)
                UA_valid.append([]), UA_test.append([]), EUC_test.append([]), COS_test.append([])

                for fold in range(n_fold):
                    fold_EUC_test, fold_COS_test, fold_UA_valid, fold_UA_test = self.train_fold(DATASET,fold)
                    UA_valid[-1] += fold_UA_valid
                    UA_test[-1] += fold_UA_test
                    EUC_test[-1] += fold_EUC_test
                    COS_test[-1] += fold_COS_test
                
                print("WC Domain [%s] valid UA: %.2f-%.4f test UA %.2f-%.4f" %(DATASET, np.mean(UA_valid[-1]),np.std(UA_valid[-1]),
                        np.mean(UA_test[-1]),np.std(UA_test[-1])))

    def test(self,my_net):
        my_net.eval()
        
        _, tmp_ua = wc_evaluation(my_net, [self.dataset[DataType.X_VALIDATION], self.dataset[DataType.X_TEST]], \
                                                [self.dataset[DataType.Y_VALIDATION],self.dataset[DataType.Y_TEST]], 0, device)
        best_UA_valid = tmp_ua[0]
        best_UA_test = tmp_ua[-1]
    
        x_eval = torch.Tensor(self.dataset[DataType.X_TEST]).to(device).cuda()
        class_output, _, _ = my_net(x_eval, alpha=0)
        class_output = F.softmax(class_output,1)
        best_EUC_test = np.sqrt(((np.array(class_output.tolist())-self.dataset[DataType.YS_TEST])**2).sum(axis=-1)).mean()
        best_COS_test = cosine_similarity(np.array(class_output.tolist()),self.dataset[DataType.YS_TEST]).diagonal().mean()

        return best_UA_valid,best_UA_test,best_EUC_test,best_COS_test

# class ScoreManager():
#     def __init__(self):
#         self.list_dict = {}

#     def init_values(self,names):
#         for name in names:
#             self.list_dict[name] = []

#     def append_by_dict(self,dict):
#         for key,value in dict.items():
#             self.list_dict[key].append(value)

#     def __getitem__(self,key):
#         return self.list_dict[key]


if __name__ == '__main__':
    x = Trainer()
    x.train()