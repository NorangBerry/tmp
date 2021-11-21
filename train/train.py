from train.train_base import Trainer

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