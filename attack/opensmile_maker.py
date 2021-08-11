import csv
import os
import pickle

import numpy as np
from utils.setting import SMILE_CONFIG_PATH, DATASET_PATH, SMILE_EXE_PATH
from tqdm import tqdm


class SmileMaker():
	def __init__(self,wav_dir,smile_dir,filter=(lambda x: True)):
		self.wav_dir = wav_dir
		#같은 상위폴더의 opensmile폴더
		self.smile_dir = smile_dir
		self.filter = filter
		if not os.path.exists(wav_dir):
			os.makedirs(wav_dir)
		if not os.path.exists(smile_dir):
			os.makedirs(smile_dir)

	def get_smile_path(self):
		return self.smile_dir

	def make_smile_csv(self):
		for root, _, files in os.walk(self.wav_dir):
			print('******** Opensmile CSV Making ***********')
			for audio_file in tqdm(files):
				audio_file:str = audio_file
				if audio_file.split('.')[-1] =='wav' and self.filter(audio_file):
					audio_path = os.path.join(root,audio_file)
					output_path = os.path.join(self.smile_dir,audio_file.replace('.wav','.csv'))
					cmd = f"{SMILE_EXE_PATH} -C {SMILE_CONFIG_PATH} -I {audio_path} -O {output_path} -noconsoleoutput 1 -nologfile 1"
					os.system(cmd)

	def parse_smile_csv(self,path):
		csvfile = open(path,'r')
		reader = [each for each in csv.reader(csvfile, delimiter=';')]
		csvfile.close()
		feats_value=str(reader[-1]).split(',')[1:-1]
		feats_name = [each[0].split(' ')[1] for each in reader[3:-5]]
		feats = list(zip(feats_name,feats_value))
		return feats_value

	def make_pickle_file(self):
		x_data = []
		file_name_data = []
		for root, subdirs, files in os.walk(self.smile_dir):
			for file in files:
				if file.split('.')[-1] == 'csv':
					data = self.parse_smile_csv(os.path.join(root,file))
					if len(data) != 0:
						x_data.append(data)
						file_name_data.append(file.replace('.csv','.wav'))
		x_data = np.array(x_data,dtype=float)
		data = {'x_data':x_data, 'file_name':file_name_data}
		filename = os.path.join(self.smile_dir,"emobase2010.pickle")
		with open(filename, 'wb') as handle:
		   pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class CREMASmileMaker(SmileMaker):
	def make_pickle_file(self):
		labels = ['neu', 'hap', 'sad', 'ang']
		x_data = []
		y_data = []
		s_data = []
		file_name_data = []
		for root, subdirs, files in os.walk(self.smile_dir):
			for file in files:
				filename,ext = (i for i in file.split('.'))
				if ext == 'csv':
					smile_data = self.parse_smile_csv(os.path.join(root,file))
					file_info_list = filename.split('_')
					label = file_info_list[2].lower()
					speaker = int(file_info_list[0])-1001
					if len(smile_data) != 0 and len(file_info_list) >= 4 and label in labels:
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
		filename = os.path.join(self.smile_dir,"emobase2010.pickle")
		with open(filename, 'wb') as handle:
		   pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def is_valid_crema(filename):
	if filename.split('_')[2] in ['NEU','HAP','SAD','ANG']:
		return True
	return False

def is_valid_noise(filename):
	return True


if __name__ == '__main__':
	maker = CREMASmileMaker(os.path.join(DATASET_PATH,"CREMA-D"),is_valid_crema)
	# maker.make_smile_csv()
	maker.make_pickle_file()