import os
from attack.noisy_sound_generator import NoisySoundGenerator
from preprocessiong.data_reader import CremaReader, MusanReader
from smile.opensmile_maker import CREMASmileMaker, IemocapSmileMaker
from tqdm import tqdm

from utils.setting import DATASET_PATH


class DataGenerator:
    def __init__(self,dataset,noise=False):
        self.dataset = dataset
        self.data_path = os.path.join(DATASET_PATH,dataset)
        self.smile_path = os.path.join(self.data_path,"opensmile")
        self.is_noise = noise
        if self.is_noise == True:
            self.noise_path = os.path.join(DATASET_PATH,"musan")

    def generate_from_one_wav(self):
        # make opensmile csv
        maker = self.get_smile_maker()
        if self.pickle_exists() == False:
            maker.make_smile_csv()
            # make pickle zipped data
            maker.make_pickle_file()
            # remove unused data

    def generate_noise_mixing_wav(self,dB):
        maker = self.get_noise_maker()
        maker.generate(os.path.join(DATASET_PATH,f"{self.dataset}_{dB}"),dB)
        # for file in tqdm(reader.get_file_list()):
        #     pass
    def get_smile_maker(self):
        maker = None
        if self.dataset == "CREMA-D":
            maker = CREMASmileMaker(self.data_path,self.smile_path)
        elif self.dataset == "IEMOCAP":
            maker = IemocapSmileMaker(self.data_path,self.smile_path)
        return maker

    def get_file_reader(self):
        reader = None
        if self.dataset == "CREMA-D":
            reader = CremaReader(self.data_path)
        return reader

    def get_noise_maker(self):
        musan_reader = MusanReader(self.noise_path)
        voice_reader = self.get_file_reader()
        maker = NoisySoundGenerator(voice_reader.get_file_list(),musan_reader.get_file_list())
        return maker

    def pickle_exists(self):
        return os.path.isfile(os.path.join(self.smile_path,"emobase2010.pickle"))
