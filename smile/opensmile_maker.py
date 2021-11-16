import abc
import csv
import os
import pickle

import numpy as np
from utils.setting import SMILE_CONFIG_PATH, DATASET_PATH, SMILE_EXE_PATH
from tqdm import tqdm


class SmileMaker():
    def __init__(self,input_dir,output_dir):
        self.input_dir = input_dir
        #같은 상위폴더의 opensmile폴더
        self.output_dir = output_dir
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def make_smile_csv(self):
        for root, _, files in os.walk(self.input_dir):
            print('******** Opensmile CSV Making ***********')
            for audio_file in tqdm(files):
                audio_file:str = audio_file

                if audio_file.split('.')[-1] !='wav' \
                or audio_file[0] != '.' \
                or self.is_valid_file(audio_file) == False:
                    continue

                audio_path = os.path.join(root,audio_file)
                output_path = os.path.join(self.output_dir,audio_file.replace('.wav','.csv'))
                cmd = f"{SMILE_EXE_PATH} -C {SMILE_CONFIG_PATH} -I {audio_path} -O {output_path} -noconsoleoutput 1 -nologfile 1"
                os.system(cmd)

    def parse_smile_csv(self,path):
        csvfile = open(path,'r')
        reader = [each for each in csv.reader(csvfile, delimiter=';')]
        csvfile.close()
        feats_value=str(reader[-1]).split(',')[1:-1]
        # feats_name = [each[0].split(' ')[1] for each in reader[3:-5]]
        # feats = list(zip(feats_name,feats_value))
        return feats_value

    @abc.abstractmethod
    def make_pickle_file(self):
        pass

    @abc.abstractmethod
    def is_valid_file(self,filename):
        pass

class CREMASmileMaker(SmileMaker):
    def make_pickle_file(self):
        labels = ['neu', 'hap', 'sad', 'ang']
        x_data,y_data,s_data,file_name_data = [],[],[],[]

        for root, _, files in os.walk(self.output_dir):
            for file in files:
                filename,ext = (i for i in file.split('.'))
                if ext == 'csv':
                    smile_data = self.parse_smile_csv(os.path.join(root,file))
                    label,speaker = self.parse_file_name(filename)
                    # file_info_list = filename.split('_')
                    # label = file_info_list[2].lower()
                    # speaker = int(file_info_list[0])-1001
                    if len(smile_data) != 0 and label in labels:
                        x_data.append(smile_data)
                        file_name_data.append(file.replace('.csv','.wav'))
                        y_data.append(labels.index(label))
                        s_data.append(speaker)

        x_data = np.array(x_data,dtype=float)
        y_data = np.array(y_data)
        data = {'x_data':x_data,
                'y_data':y_data,
                's_data':s_data,
                # 'ys_data':ys_data,
                'file_name':file_name_data}
        filename = os.path.join(self.output_dir,"emobase2010.pickle")
        with open(filename, 'wb') as handle:
           pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def parse_file_name(self,filename):
        file_info_list = filename.split('_')
        label = file_info_list[2].lower()
        speaker = int(file_info_list[0])-1001
        return label,speaker

    def is_valid_file(self,filename):
        if filename.split('_')[2] in ['NEU','HAP','SAD','ANG']:
            return True
        return False

class IemocapSmileMaker(SmileMaker):
    def make_pickle_file(self):
        x_data = []
        file_name_data = []
        for root, _, files in os.walk(self.output_dir):
            for file in files:
                file:str = file
                if file.split('.')[-1] != 'csv':
                    continue
                
                data = self.parse_smile_csv(os.path.join(root,file))
                if len(data) == 0:
                    continue
                
                x_data.append(data)
                file_name_data.append(file.replace('.csv','.wav'))

        x_data = np.array(x_data,dtype=float)
        data = {'x_data':x_data, 'file_name':file_name_data}
        
        filename = os.path.join(self.output_dir,"emobase2010.pickle")
        with open(filename, 'wb') as handle:
           pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def is_valid_file(self,filename):
        if filename.split('_')[2] in ['NEU','HAP','SAD','ANG']:
            return True
        return False


# def is_valid_noise(filename):
#     return True


# if __name__ == '__main__':
#     maker = CREMASmileMaker(os.path.join(DATASET_PATH,"CREMA-D"),is_valid_crema)
#     # maker.make_smile_csv()
#     maker.make_pickle_file()