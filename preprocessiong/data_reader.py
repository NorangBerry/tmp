import os
from tqdm import tqdm

class CremaReader:
    def __init__(self, data_path):
        self.data_path = data_path
    def get_file_list(self):
        file_list = []
        for root, _, files in os.walk(self.data_path):
            for audio_file in tqdm(files):
                audio_file:str = audio_file

                if audio_file.split('.')[-1] !='wav' \
                or audio_file[0] == '.' \
                or self.is_valid_file(audio_file) == False:
                    continue
                file_list.append(os.path.join(root,audio_file))
        return file_list

    def is_valid_file(self,filename):
        if filename.split('_')[2] in ['NEU','HAP','SAD','ANG']:
            return True
        return False

class MusanReader:
    def __init__(self, data_path):
        self.data_path = data_path
    def get_file_list(self):
        file_list = []
        for root, _, files in os.walk(self.data_path):
            for audio_file in tqdm(files):
                audio_file:str = audio_file

                if audio_file.split('.')[-1] !='wav' \
                or audio_file[0] == '.' \
                or self.is_valid_file(audio_file) == False:
                    continue
                file_list.append(os.path.join(root,audio_file))
        return file_list

    def is_valid_file(self,filename):
        return True
