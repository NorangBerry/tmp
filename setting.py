from pathlib import Path
import os
ROOT_PATH = Path(__file__).resolve().parent.resolve().parent
DATASET_PATH = "C:/Users/sapl_Junyoung/Desktop/dataset"
SMILE_PATH = os.path.join(ROOT_PATH, "opensmile")
CONFIG = "emobase/emobase2010.conf"
DATA_PATH = os.path.join(ROOT_PATH,"CREMA-D","emobase2010")
MODEL_PATH = os.path.join(ROOT_PATH,"CREMA-D","emobase2010","WC0716_JY")

class Setting():
    def __init__(self,dict = {}):
        for key in dict:
            setattr(self, key, dict[key])