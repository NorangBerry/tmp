import os
from smile.opensmile_maker import CREMASmileMaker, IemocapSmileMaker

from utils.setting import DATASET_PATH


class DataGenerator:
    def __init__(self,dataset):
        self.dataset = dataset
        self.data_path = os.path.join(DATASET_PATH,dataset)
        self.smile_path = os.path.join(self.data_path,"opensmile")

    def generate_from_one_wav(self):
        # make opensmile csv
        maker = self.get_smile_maker()
        if self.pickle_exists() == False:
            maker.make_smile_csv()
            # make pickle zipped data
            maker.make_pickle_file()
            # remove unused data

    def get_smile_maker(self):
        maker = None
        if self.dataset == "CREMA-D":
            maker = CREMASmileMaker(self.data_path,self.smile_path)
        elif self.dataset == "IEMOCAP":
            maker = IemocapSmileMaker(self.data_path,self.smile_path)
        return maker

    def pickle_exists(self):
        return os.path.isfile(os.path.join(self.smile_path,"emobase2010.pickle"))
