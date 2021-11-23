from train.base import DataType, ModelRunner
from utils.setting import device, get_pickle_path
from utils.data_loader import load_emotion_corpus_WC
import os 
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import numpy as np
import torch
from utils.functions import normalization_ops, wc_evaluation

class Tester(ModelRunner):
    def __init__(self,train_dataset,test_dataset,test_fold):
        super().__init__(train_dataset)
        self.trainDB:str = train_dataset
        self.test_fold:int = test_fold
        self.test_dataset:str = test_dataset
        self.data_path:str = get_pickle_path(self.test_dataset)
        self.test_result = {}

    def set_data(self,fold):
        dataset = self.test_dataset.split('_')[0]
        x_train, y_train, x_valid, y_valid, x_test, y_test, ys_test = load_emotion_corpus_WC(dataset, self.data_path, self.test_fold)
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
        tokens = self.trainDB.split('_')
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
        elif len(tokens) >= 2 and tokens[1] == "noisy":
            info["NoiseType"] = "noisy"
            info["dB"] = f"{tokens[2]}dB"
        elif len(tokens) >= 2 and tokens[1] == "gradient":
            info["NoiseType"] = "gradient"
            info["gradient"] = f"0.{tokens[2]}"
        return info