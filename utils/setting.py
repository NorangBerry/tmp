from pathlib import Path
import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DATASET_LIST = ['CREMA-D'] # ['DEMoS']# ['CREMA-D','IEMOCAP','MSPIMPROV']

ROOT_PATH = "D:/GIST/deep"#Path(__file__).resolve().parent.resolve().parent

DATASET_PATH = "D:/GIST/deep"

SMILE_PATH = os.path.join(ROOT_PATH, "opensmile")

SMILE_EXE_PATH = os.path.join(SMILE_PATH,'bin','SMILExtract')
SMILE_CONFIG_PATH = os.path.join(SMILE_PATH,"config", "emobase/emobase2010.conf")

CREATED_WAV_DIR = os.path.join(ROOT_PATH, "created")

def get_model_dir(dataset,type=None,value=None,oversample = False):
    if oversample == True:
        return os.path.join(get_dataset_folder(dataset,type,value),"emobase2010","WC1128_JY")    
    return os.path.join(get_dataset_folder(dataset,type,value),"emobase2010","WC0802_JY")

def get_dataset_folder(dataset,type=None,value=None):
    if type == "gradient":
        value = f"{value:.2f}"[-2:]
    if type == None:
        return os.path.join(ROOT_PATH,dataset)
    else:
        return os.path.join(ROOT_PATH,f"{dataset}_{type}_{value}")

def get_pickle_path(dataset,type=None,value=None):
    return os.path.join(get_dataset_folder(dataset,type,value),"opensmile","emobase2010")

class Setting():
    def __init__(self,dict = {}):
        for key in dict:
            setattr(self, key, dict[key])

crema_setting = Setting(dict={
            "n_epochs" : 100,
            "lr" : 2e-4,
            "batch_size" : 32,
            "n_patience" : 5,
            "n_seeds" : 5
        })