
from tqdm import tqdm
from checker import Tester
import csv
from noise_combine import Noise_Combiner
import pickle
from utils.functions import normalization_ops, wLoss
from utils.setting import DATASET_PATH, ROOT_PATH, get_model_dir
import torch
import os 
import numpy as np 
import re
import fnmatch
import random

DATA_PATH = os.path.join(ROOT_PATH,"my_crema","opensmile")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_pth_files(path):
	return fnmatch.filter(os.listdir(path),'*.pth')

class Attacker():
	def __init__(self):
		self.data_path = DATA_PATH
		self.x_data,self.y_data,self.x_filenames = self.load_data()
		self.testers:'list[Tester]' = []
		self.noise_maker = Noise_Combiner()

		model_dir = get_model_dir("CREMA-D")
		pre_trained_models = get_pth_files(model_dir)
		for model_path in pre_trained_models:
			self.testers.append(Tester(torch.load(os.path.join(model_dir,model_path)),
									wLoss().cuda()))

	def load_data(self) -> 'tuple[list[np.ndarray],list[np.ndarray]]':
		train_filename = os.path.join(self.data_path,"emobase2010.pickle")
		with open(train_filename, 'rb') as handle:
			data = pickle.load(handle)
			x_data = data['x_data']
			y_data = data['y_data']
			x_filenames = data['file_name']
			feat_mu = np.mean(x_data,axis=0)
			feat_st = np.std(x_data, axis=0)

			x_data  = normalization_ops(feat_mu, feat_st, x_data)
			return x_data, y_data, x_filenames

	def attack_file_generate(self):
		big_feature_index_map = self.get_smile_big_feature_index_map()
		for x_data,filename in tqdm(zip(self.x_data,self.x_filenames),total=len(self.x_data)):
			noise_file_name = self.noise_maker.get_max_diff_noise(x_data,big_feature_index_map["minPos"])

			voice_file_path = os.path.join(DATASET_PATH,"CREMA-D",filename)
			noise_file_path = os.path.join(DATASET_PATH,"musan","noise")
			if 'free' in noise_file_name:
				noise_file_path = os.path.join(noise_file_path,"free-sound",noise_file_name)
			else:
				noise_file_path = os.path.join(noise_file_path,"sound-bible",noise_file_name)
			save_path = self.noise_maker.combine_noise(voice_file_path,noise_file_path)

		self.noise_maker.extract_opensmile_csv()
		self.noise_maker.make_pickle_file()

	def sample_attack_generate(self):
		noise_maker = Noise_Combiner(os.path.join(ROOT_PATH,"random_test"))
		noise_file_names = noise_maker.noise_dataset['file_name']
		for x_data,filename in tqdm(zip(self.x_data,self.x_filenames),total=len(self.x_data)):
			noise_file_name = random.choice(noise_file_names)
			voice_file_path = os.path.join(DATASET_PATH,"CREMA-D",filename)
			noise_file_path = os.path.join(DATASET_PATH,"musan","noise")
			if 'free' in noise_file_name:
				noise_file_path = os.path.join(noise_file_path,"free-sound",noise_file_name)
			else:
				noise_file_path = os.path.join(noise_file_path,"sound-bible",noise_file_name)
			save_path = noise_maker.combine_noise(voice_file_path,noise_file_path)
		noise_maker.extract_opensmile_csv()
		noise_maker.make_pickle_file()
			

	def test_attack(self):
		for tester in self.testers:
			x_data = torch.Tensor(self.x_data).cuda()
			y_data = torch.Tensor(self.y_data).unsqueeze(1).cuda()
			print(tester.test(x_data,y_data))
		noise_maker = Noise_Combiner(os.path.join(ROOT_PATH,"random_test"))
		for tester in self.testers:
			dataset = noise_maker.load_combined_voice()
			x_data = torch.Tensor(dataset['x_data']).cuda()
			y_data = torch.Tensor(dataset['y_data']).unsqueeze(1).cuda()
			print(len(x_data))
			print(tester.test(x_data,y_data))
		for tester in self.testers:
			dataset = self.noise_maker.load_combined_voice()
			x_data = torch.Tensor(dataset['x_data']).cuda()
			y_data = torch.Tensor(dataset['y_data']).unsqueeze(1).cuda()
			print(len(x_data))
			print(tester.test(x_data,y_data))

	# FGSM 공격 코드
	def fgsm_attack(self,data, epsilon, data_grad,clip_range=None) -> torch.Tensor:
		# data_grad 의 요소별 부호 값을 얻어옵니다
		sign_data_grad = -1 * data_grad.sign()
		# 입력 이미지의 각 픽셀에 sign_data_grad 를 적용해 작은 변화가 적용된 이미지를 생성합니다
		perturbed_data = data + epsilon*sign_data_grad

		# 값 범위를 normalize 범위로 유지하기 위해 자르기(clipping)를 추가합니다
		if clip_range is not None:
			perturbed_data = torch.clamp(perturbed_data, clip_range[0], clip_range[1])

		# 작은 변화가 적용된 이미지를 리턴합니다
		return perturbed_data.to(device).cuda()

	def feature_attack(self,data,epsilon,data_grad,feature_idx) -> torch.Tensor:
		# data_grad 의 요소별 부호 값을 얻어옵니다
		sign_data_grad = -1 * data_grad.sign()
		# 입력 이미지의 각 픽셀에 sign_data_grad 를 적용해 작은 변화가 적용된 이미지를 생성합니다
		perturbed_data = data.clone()
		perturbed_data[0][feature_idx] += epsilon*sign_data_grad[0][feature_idx]

		del sign_data_grad
		# 작은 변화가 적용된 이미지를 리턴합니다
		return perturbed_data

	def big_feature_attack(self,data,epsilon,data_grad,index_list) -> torch.Tensor:
		# data_grad 의 요소별 부호 값을 얻어옵니다
		sign_data_grad = -1 * data_grad.sign()
		# 입력 이미지의 각 픽셀에 sign_data_grad 를 적용해 작은 변화가 적용된 이미지를 생성합니다
		perturbed_data = data.clone()
		perturbed_data[0][index_list] += epsilon*sign_data_grad[0][index_list]

		del sign_data_grad
		# 작은 변화가 적용된 이미지를 리턴합니다
		return perturbed_data

	def get_smile_big_feature_index_map(self) -> 'dict[str,list[int]]':
		feature_names = self.load_smile_feature_names()
		feature_names:'list[list[str]]' = [feature_name.split('_') for feature_name in feature_names]
		ret:'dict[str,list[int]]' = {}
		for i, sub_features in enumerate(feature_names):
			for sub_feature in sub_features:
				sub_feature = re.sub(r'\[[1-9]+\]', '', sub_feature)
				if '' == sub_feature:
					continue
				if sub_feature not in ret.keys():
					ret[sub_feature] = []
				ret[sub_feature].append(i)
		return ret

	def load_smile_feature_names(self) -> 'list[str]':
		path = os.path.join(DATA_PATH,"opensmile_props.csv")
		csvfile = open(path,'r')
		reader = [each for each in csv.reader(csvfile, delimiter=';')]
		csvfile.close()
		feats_name = [each[0].split(' ')[1] for each in reader[3:-5]]
		return feats_name

if __name__ == '__main__':
	attacker = Attacker()
	# attacker.sample_attack_generate()
	attacker.test_attack()