from utils.setting import DATASET_LIST
from train.train import Trainer

for dataset in DATASET_LIST:
	x = Trainer(dataset)
	x.train()