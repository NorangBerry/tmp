import os
from attack.noisy_sound_generator import NoisySoundGenerator
from preprocessiong.data_reader import CremaReader, IemocapReader, MusanReader
from smile.opensmile_maker import CREMASmileMaker, IemocapSmileMaker, SmileMaker
from tqdm import tqdm

from utils.setting import DATASET_PATH


class DataGenerator:
    def __init__(self,dataset,noise=False):
        self.dataset = dataset
        self.data_path = os.path.join(DATASET_PATH,dataset)
        self.generate_target_path = self.data_path
        self.is_noise = noise
        if self.is_noise == True:
            self.noise_path = os.path.join(DATASET_PATH,"musan")

    def generate_from_one_wav(self):
        # make opensmile csv
        maker:SmileMaker = self.get_smile_maker()
        if self.pickle_exists() == True:
            return
        maker.make_smile_csv()
        # make pickle zipped data
        maker.make_pickle_file()
        # remove unused data

    def generate_noise_mixing_wav(self,dB):
        self.generate_target_path = os.path.join(DATASET_PATH,f"{self.dataset}_{dB}")

        if self.pickle_exists() == True:
            return
        if os.path.isdir(self.generate_target_path) == False:
            noise_maker = self.get_noise_maker()
            noise_maker.generate(self.generate_target_path,dB)


        smile_maker = self.get_smile_maker()
        smile_maker.make_smile_csv()
        # make pickle zipped data
        smile_maker.make_pickle_file()
        # for file in tqdm(reader.get_file_list()):
        #     pass


    def get_smile_maker(self) -> SmileMaker:
        maker = None
        if self.dataset == "CREMA-D":
            maker = CREMASmileMaker(self.generate_target_path,
                os.path.join(self.generate_target_path,"opensmile"))
        elif self.dataset == "IEMOCAP":
            maker = IemocapSmileMaker(self.generate_target_path,
                os.path.join(self.generate_target_path,"opensmile"),os.path.join(self.data_path))
        return maker

    def get_file_reader(self):
        reader = None
        if self.dataset == "CREMA-D":
            reader = CremaReader(self.data_path)
        elif self.dataset == "IEMOCAP":
            reader = IemocapReader(self.data_path)
        return reader

    def get_noise_maker(self):
        musan_reader = MusanReader(self.noise_path)
        voice_reader = self.get_file_reader()
        maker = NoisySoundGenerator(voice_reader.get_file_list(),musan_reader.get_file_list())
        return maker

    def pickle_exists(self):
        return os.path.isfile(os.path.join(self.generate_target_path,"opensmile","emobase2010.pickle"))
