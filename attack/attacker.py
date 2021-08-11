
import random
from tqdm import tqdm
from .checker import Tester
import csv
from .noise_combine import Noise_Combiner
import pickle
from utils.functions import normalization_ops, wLoss
from utils.setting import DATASET_PATH, ROOT_PATH, get_model_dir
import torch
import os 
import numpy as np 
import re
import fnmatch
import statistics
from collections import Counter
DATA_PATH = os.path.join(ROOT_PATH,"my_crema","opensmile")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_pth_files(path):
	return fnmatch.filter(os.listdir(path),'*.pth')

class Attacker():
	def __init__(self,save_path):
		self.voice_pickle_path = os.path.join(ROOT_PATH,"CREMA-D","opensmile","emobase2010.pickle")
		self.noise_data_path = os.path.join(DATASET_PATH,"musan","noise")
		self.voice_data_path = os.path.join(DATASET_PATH,"CREMA-D")

		self.voice_data,self.voice_label,self.voice_filenames = self.load_voice_data()

		self.noise_maker = Noise_Combiner(save_path)

		self.testers:'list[Tester]' = []
		model_dir = get_model_dir("CREMA-D")
		pre_trained_models = get_pth_files(model_dir)
		for model_path in pre_trained_models:
			self.testers.append(Tester(torch.load(os.path.join(model_dir,model_path)),
									wLoss().cuda()))

	def load_voice_data(self) -> 'tuple[list[np.ndarray],list[np.ndarray]]':
		with open(self.voice_pickle_path, 'rb') as handle:
			data = pickle.load(handle)
			x_data = data['x_data']
			y_data = data['y_data']
			x_filenames = data['file_name']
			feat_mu = np.mean(x_data,axis=0)
			feat_st = np.std(x_data, axis=0)

			x_data  = normalization_ops(feat_mu, feat_st, x_data)
			return x_data, y_data, x_filenames

	def attack_file_generate(self,option=None) -> None:
		pass

	def is_pickle_exist(self):
		pickle_path = os.path.join(self.noise_maker.opensmile_manager.get_smile_path(),"emobase2010.pickle")
		return os.path.exists(pickle_path)

	def test_attack(self,option=None):
		if self.is_pickle_exist() == False:
			self.attack_file_generate(option)
		accuracy_list = []
		for tester in self.testers:
			dataset = self.noise_maker.load_combined_voice()
			x_data = torch.Tensor(dataset['x_data']).cuda()
			y_data = torch.Tensor(dataset['y_data']).unsqueeze(1).cuda()
			accuracy = tester.test(x_data,y_data)
			print(accuracy)
			accuracy_list.append(accuracy)
		return statistics.mean(accuracy_list)

	def get_smile_big_feature_index_map(self) -> 'dict[str,list[int]]':
		feature_names = self.load_smile_feature_names()
		feature_names:'list[list[str]]' = [feature_name.split('_') for feature_name in feature_names]
		ret:'dict[str,list[int]]' = {}
		for i, sub_features in enumerate(feature_names):
			for sub_feature in sub_features:
				sub_feature = re.sub(r'\[[0-9]+\]', '', sub_feature)
				if '' == sub_feature:
					continue
				while sub_feature[-1] in ['1','2','3','4','5','6','7','8','9','0','-']:
					sub_feature = sub_feature[:-1]
				if sub_feature not in ret.keys():
					ret[sub_feature] = []
				ret[sub_feature].append(i)
		return ret

	def load_smile_feature_names(self) -> 'list[str]':
		path = os.path.join(ROOT_PATH,"opensmile_props.csv")
		csvfile = open(path,'r')
		reader = [each for each in csv.reader(csvfile, delimiter=';')]
		csvfile.close()
		feats_name = [each[0].split(' ')[1] for each in reader[3:-5]]
		return feats_name

	def get_most_used_noise(self,count:int) -> 'list[str]':
		dataset = self.noise_maker.load_combined_voice()
		noise_file_list:'list[str]' = dataset['file_name']
		noise_file_list = [noise_file_name.split('_')[-1] for noise_file_name in noise_file_list]
		# print(noise_file_list)
		noise_file_counter = Counter(noise_file_list).most_common(count)
		top_noise_files = [key for key,cnt in noise_file_counter]
		return top_noise_files

class FeatureAttacker(Attacker):
	def attack_file_generate(self,feature) -> None:
		big_feature_index_map = self.get_smile_big_feature_index_map()
		print('******** Noise WAV Making ***********')
		for voice_data,filename in tqdm(zip(self.voice_data,self.voice_filenames),total=len(self.voice_data)):
			noise_file_name = self.noise_maker.get_max_diff_noise(voice_data,big_feature_index_map[feature],feature)

			voice_file_path = os.path.join(self.voice_data_path,filename)
			noise_file_path = ""
			if 'free' in noise_file_name:
				noise_file_path = os.path.join(self.noise_data_path,"free-sound",noise_file_name)
			else:
				noise_file_path = os.path.join(self.noise_data_path,"sound-bible",noise_file_name)
			self.noise_maker.combine_noise(voice_file_path,noise_file_path)

		self.noise_maker.extract_opensmile_csv()
		self.noise_maker.make_pickle_file()
		self.noise_maker.remove_generated_files()

class RandomAttacker(Attacker):
	def attack_file_generate(self,option=None) -> None:
		noise_file_names = self.noise_maker.noise_dataset['file_name']
		print('******** Noise WAV Making ***********')
		for filename in tqdm(self.voice_filenames,total=len(self.voice_data)):
			for i in range(10):
				noise_file_name = random.choice(noise_file_names)
				voice_file_path = os.path.join(self.voice_data_path,filename)
				noise_file_path = ""
				if 'free' in noise_file_name:
					noise_file_path = os.path.join(self.noise_data_path,"free-sound",noise_file_name)
				else:
					noise_file_path = os.path.join(self.noise_data_path,"sound-bible",noise_file_name)
				self.noise_maker.combine_noise(voice_file_path,noise_file_path)

		self.noise_maker.extract_opensmile_csv()
		self.noise_maker.make_pickle_file()
		self.noise_maker.remove_generated_files()

class FixedNoiseAttacker(Attacker):
	def attack_file_generate(self,noise_file_name) -> None:
		print('******** Noise WAV Making ***********')
		for filename in tqdm(self.voice_filenames,total=len(self.voice_data)):
			voice_file_path = os.path.join(self.voice_data_path,filename)
			noise_file_path = ""
			if 'free' in noise_file_name:
				noise_file_path = os.path.join(self.noise_data_path,"free-sound",noise_file_name)
			else:
				noise_file_path = os.path.join(self.noise_data_path,"sound-bible",noise_file_name)
			self.noise_maker.combine_noise(voice_file_path,noise_file_path)

		self.noise_maker.extract_opensmile_csv()
		self.noise_maker.make_pickle_file()
		self.noise_maker.remove_generated_files()