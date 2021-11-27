import os
from train.base import DataType
from train.train_base import FgsmTrainer, NoiseTrainer, Trainer
from utils.data_loader import load_emotion_corpus_WC
from utils.functions import karyogram, normalization_ops
import numpy as np

from utils.setting import get_pickle_path
class CremaTrainer(Trainer):
    def __init__(self):
        super().__init__("CREMA-D")

    def get_n_fold(self):
        return 1

class CremaNoiseTrainer(Trainer):
    def __init__(self,dB):
        super().__init__("CREMA-D")
        self.noise_path = get_pickle_path(self.dataset,"noisy",dB)


    def set_data(self,fold):
        karyogram()
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

    def get_n_fold(self):
        return 1

class IemocapTrainer(Trainer):
    def __init__(self):
        super().__init__("IEMOCAP")
    def get_n_fold(self):
        return 10

class CremaNoiseTrainer(NoiseTrainer):
    def __init__(self,dB):
        super().__init__("CREMA-D",dB)
        self.noise_path = get_pickle_path(self.dataset,"noisy",dB)
    def get_n_fold(self):
        return 1

class IemocapNoiseTrainer(NoiseTrainer):
    def __init__(self,dB):
        super().__init__("IEMOCAP",dB)
        self.noise_path = get_pickle_path(self.dataset,"noisy",dB)
    def get_n_fold(self):
        return 10


class CremaFgsmTrainer(FgsmTrainer):
    def __init__(self,epsilon):
        super().__init__("CREMA-D",epsilon)
        self.noise_path = get_pickle_path(self.dataset,"gradient",epsilon)
    def get_n_fold(self):
        return 1

class IemocapFgsmTrainer(FgsmTrainer):
    def __init__(self,epsilon):
        super().__init__("IEMOCAP",epsilon)
        self.noise_path = get_pickle_path(self.dataset,"gradient",epsilon)
    def get_n_fold(self):
        return 10