import csv
import os
import pickle

import numpy as np
from setting import ROOT_PATH

SMILE_PATH = os.path.join(ROOT_PATH, "opensmile")
CONFIG = "emobase/emobase2010.conf"

class SmileMaker():
	def __init__(self,target_dir,save_dir,filter=(lambda x: True)):
		self.save_path = os.path.join(ROOT_PATH, save_dir,"opensmile")
		self.conf_path = os.path.join(SMILE_PATH,"config", CONFIG)
		self.exe_path = os.path.join(SMILE_PATH,'bin','SMILExtract')
		self.target_dir = target_dir
		self.filter = filter

	def make_smile_csv(self):
		if not os.path.exists(self.save_path):
				os.makedirs(self.save_path)
		for root, _, files in os.walk(self.target_dir):
			for audio_file in files:
				if audio_file.split('.')[-1] =='wav' and self.filter(audio_file):
					audio_path = os.path.join(root,audio_file)
					output_path = os.path.join(self.save_path,audio_file.replace('.wav','.csv'))
					cmd = f"{self.exe_path} -C {self.conf_path} -I {audio_path} -O {output_path}"
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
		for root, subdirs, files in os.walk(self.save_path):
			for file in files:
				if file.split('.')[-1] == 'csv':
					x_data.append(self.parse_smile_csv(os.path.join(root,file)))
		data = {'x_data':x_data}
		filename = os.path.join(self.save_path,"emobase2010.pickle")
		with open(filename, 'wb') as handle:
		   pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def is_valid_crema(filename):
	if filename.split('_')[2] in ['SAD','HAP','ANG','NEU']:
		return True
	return False

def is_valid_noise(filename):
	return True

if __name__ == '__main__':
	pass
	# noise_pickle_maker = SmileMaker(
	# 	target_dir="C:/Users/sapl_Junyoung/Desktop/dataset/musan/noise",
	# 	save_dir=os.path.join(ROOT_PATH, "musan","opensmile"),
	# 	filter=is_valid_noise
	# )
	# noise_pickle_maker.make_smile_csv()
	# noise_pickle_maker.make_pickle_file()

	# noise_pickle_maker = SmileMaker(
	# 	target_dir="C:/Users/sapl_Junyoung/Desktop/dataset/CREMA-D",
	# 	save_dir=os.path.join(ROOT_PATH, "CREMA-D","opensmile"),
	# 	filter=is_valid_crema
	# )
	# noise_pickle_maker.make_smile_csv()
	# noise_pickle_maker.make_pickle_file()