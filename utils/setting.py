from pathlib import Path
import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DATASET_LIST = ['CREMA-D'] # ['DEMoS']# ['CREMA-D','IEMOCAP','MSPIMPROV']

ROOT_PATH = "C:/codes/SER_base/210716_WC-CRE_for_JunyoungChoi"#Path(__file__).resolve().parent.resolve().parent

DATASET_PATH = "C:/Users/sapl_Junyoung/Desktop/dataset"

SMILE_PATH = os.path.join(ROOT_PATH, "opensmile")

SMILE_EXE_PATH = os.path.join(SMILE_PATH,'bin','SMILExtract')
SMILE_CONFIG_PATH = os.path.join(SMILE_PATH,"config", "emobase/emobase2010.conf")

CREATED_WAV_DIR = os.path.join(ROOT_PATH, "created")

def get_model_dir(dataset):
    return os.path.join(ROOT_PATH,dataset,"emobase2010","WC0802_JY")
    
def get_data_dir(dataset):
    return os.path.join(ROOT_PATH,dataset,"emobase2010","WC0802_JY")

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