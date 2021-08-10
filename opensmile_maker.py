import csv
import os
import pickle

import numpy as np
from setting import CONFIG, DATASET_PATH, ROOT_PATH, SMILE_PATH



class SmileMaker():
	def __init__(self,target_dir,save_dir,filter=(lambda x: True)):
		self.conf_path = os.path.join(SMILE_PATH,"config", CONFIG)
		self.exe_path = os.path.join(SMILE_PATH,'bin','SMILExtract')

		self.save_path = save_dir
		self.target_dir = target_dir
		self.filter = filter

	def make_smile_csv(self):
		if not os.path.exists(self.save_path):
				os.makedirs(self.save_path)
		for root, _, files in os.walk(self.target_dir):
			for audio_file in files:
				audio_file:str = audio_file
				if audio_file.split('.')[-1] =='wav' and self.filter(audio_file):
					audio_path = os.path.join(root,audio_file)
					output_path = os.path.join(self.save_path,audio_file.replace('.wav','.csv'))
					cmd = f"{self.exe_path} -C {self.conf_path} -I {audio_path} -O {output_path} -noconsoleoutput 1 -nologfile 1"
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
		for root, subdirs, files in os.walk(self.save_path):
			for file in files:
				if file.split('.')[-1] == 'csv':
					data = self.parse_smile_csv(os.path.join(root,file))
					if len(data) != 0:
						x_data.append(data)
						file_name_data.append(file.replace('.csv','.wav'))
		x_data = np.array(x_data,dtype=float)
		data = {'x_data':x_data, 'file_name':file_name_data}
		filename = os.path.join(self.save_path,"emobase2010.pickle")
		with open(filename, 'wb') as handle:
		   pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class CREMASmileMaker(SmileMaker):
	def make_pickle_file(self):
		labels = ['neu', 'hap', 'sad', 'ang']
		x_data = []
		y_data = []
		s_data = []
		file_name_data = []
		for root, subdirs, files in os.walk(self.save_path):
			for file in files:
				if file.split('.')[-1] == 'csv':
					data = self.parse_smile_csv(os.path.join(root,file))
					if len(data) != 0 and len(file.split('_')) >= 4:
						x_data.append(data)
						file_name_data.append(file.replace('.csv','.wav'))
						y_data.append(labels.index(file.split('_')[2].lower()))
						s_data.append(int(file.split('_')[0])-1001)
		x_data = np.array(x_data,dtype=float)
		y_data = np.array(y_data)
		data = {'x_data':x_data,
				'y_data':y_data,
				's_data':s_data,
				# 'ys_data':ys_data,
				'file_name':file_name_data}
		filename = os.path.join(self.save_path,"emobase2010.pickle")
		with open(filename, 'wb') as handle:
		   pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def is_valid_crema(filename):
	if filename.split('_')[2] in ['NEU','HAP','SAD','ANG']:
		return True
	return False

def is_valid_noise(filename):
	return True


if __name__ == '__main__':
	maker = CREMASmileMaker(os.path.join(DATASET_PATH,"CREMA-D"),os.path.join(ROOT_PATH, "my_crema","opensmile"),is_valid_crema)
	# maker.make_smile_csv()
	maker.make_pickle_file()