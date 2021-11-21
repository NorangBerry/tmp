import abc
from utils.setting import crema_setting,ROOT_PATH,device, get_model_dir
from utils.data_loader import load_emotion_corpus_WC
import random
import os 
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import torch
import random
from sklearn.metrics.pairwise import cosine_similarity
from utils.functions import normalization_ops, wc_evaluation, wLoss, makedirs
from .model import BaseModel
from enum import Enum

class DataType(Enum):
    X_TRAIN = 1
    X_TEST = 2
    X_VALIDATION = 3
    Y_TRAIN = 4
    Y_TEST = 5
    Y_VALIDATION = 6
    YS_TEST = 7

class ModelRunner():
    def __init__(self,dataset):
        self.init_setting()
        self.dataset = dataset
        self.model_dir = get_model_dir(self.dataset)
        self.data_path = os.path.join(ROOT_PATH,self.dataset,"opensmile","emobase2010")

    def init_setting(self):
        self.setting = crema_setting

    def set_random_seed(self,seed):
        print('SEED %d start' %(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @abc.abstractmethod
    def get_n_fold(self):
        pass

    def init_network(self):
        self.model = BaseModel().cuda()
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.setting.lr)
        self.my_loss   = wLoss().cuda() #FocalLoss().cuda() #
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='max', patience=2) 
   
    @abc.abstractmethod
    def set_data(self,fold):
        pass

    def run(self): 
        UA_valid, UA_test = ([] for _ in range(2))

        n_fold = self.get_n_fold()
        UA_valid.append([]), UA_test.append([])

        for fold in range(n_fold):
            fold_UA_valid, fold_UA_test = self.run_fold(fold)
            UA_valid[-1] += fold_UA_valid
            UA_test[-1] += fold_UA_test
        
        print("WC Domain [%s] valid UA: %.2f-%.4f test UA %.2f-%.4f" %(self.dataset, np.mean(UA_valid[-1]),np.std(UA_valid[-1]),
                np.mean(UA_test[-1]),np.std(UA_test[-1])))
        return UA_valid[-1],UA_test[-1]

    def run_fold(self,fold):
        ls_train,n_minibatch,tr_n_samples = self.set_data(fold)
        fold_UA_valid,fold_UA_test = ([] for _ in range(2))

        for seed in range(self.setting.n_seeds):
            best_UA_valid = 0.
            best_UA_test  = 0.
            best_UA_valid, best_UA_test = self.run_seed(seed,ls_train,tr_n_samples,n_minibatch,fold)
            fold_UA_valid.append([best_UA_valid])
            fold_UA_test.append([best_UA_test])
        return fold_UA_valid, fold_UA_test

    @abc.abstractmethod
    def run_seed(self,seed,ls_train,tr_n_samples,n_minibatch,fold):
        pass


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

class Tester(ModelRunner):
    def __init__(self,train_dataset,test_dataset,test_fold):
        super().__init__(train_dataset)
        self.trainDB:str = train_dataset
        self.test_fold:int = test_fold
        self.test_dataset:str = test_dataset
        self.data_path:str = os.path.join(ROOT_PATH,self.test_dataset,"opensmile","emobase2010")
        self.test_result = {}

    def set_data(self,fold):
        x_train, y_train, x_valid, y_valid, x_test, y_test, ys_test = load_emotion_corpus_WC(self.test_dataset, self.data_path, self.test_fold)
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

    def run_seed(self,seed,ls_train,tr_n_samples,n_minibatch,fold):
        my_net = torch.load(os.path.join(self.model_dir,f"WC_fold{fold}_seed{seed}.pth"))
        my_net.eval()
        _, tmp_ua = wc_evaluation(my_net, [self.dataloader[DataType.X_VALIDATION], self.dataloader[DataType.X_TEST]], \
                                                [self.dataloader[DataType.Y_VALIDATION],self.dataloader[DataType.Y_TEST]], 0, device)
        best_UA_valid = tmp_ua[0]
        best_UA_test = tmp_ua[-1]
    
        return best_UA_valid,best_UA_test

    def run(self):
        _, test_result = super().run()
        self.test_result = {
            "Accuracy":np.mean(test_result)
        }

    def get_result(self) -> dict:
        tokens = self.trainDB.split('-')
        train_set = self.__parse_dataset_folder_info(self.trainDB)
        test_set = self.__parse_dataset_folder_info(self.test_dataset)
        test_set["Fold"] = self.test_fold
        train_set["BaseDB"] = tokens[0]
        return {
            "TrainSet": train_set,
            "TestSet": test_set,
            "Model":"DANN",
            "Result": self.test_result
        }
        
    def __parse_dataset_folder_info(self,folder:str):
        tokens = folder.split('_')
        info = {
                "BaseDB":None,
                "NoiseType":None,
        }
        info["BaseDB"] = tokens[0]
        if len(tokens) == 1:
            info["NoiseType"] = "clean"
        elif len(tokens) > 2 and tokens[1] == "noisy":
            info["NoiseType"] = "noisy"
            info["dB"] = f"{tokens[2]}dB"
        elif len(tokens) > 2 and tokens[1] == "gradient":
            info["NoiseType"] = "gradient"
            info["gradient"] = f"0.{tokens[2]}"
        return info

class CremaTrainer(Trainer):
    def __init__(self):
        super().__init__("CREMA-D")

    def get_n_fold(self):
        return 1

class IemocapTrainer(Trainer):
    def __init__(self):
        super().__init__("IEMOCAP")
    def get_n_fold(self):
        return 10

class CremaTester(Tester):
    def __init__(self,testDB,fold):
        super().__init__("CREMA-D",testDB,fold)
    def get_n_fold(self):
        return 1


class IemocapTester(Tester):
    def __init__(self,testDB,fold):
        super().__init__("IEMOCAP",testDB,fold)
    def get_n_fold(self):
        return 10