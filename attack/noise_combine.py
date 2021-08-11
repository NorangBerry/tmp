from utils.functions import normalization_ops
import pickle
from utils.setting import DATASET_PATH, ROOT_PATH
from pydub import AudioSegment
import numpy as np
import os
from opensmile_maker import CREMASmileMaker, SmileMaker

class Noise_Combiner():
	def __init__(self,dir=ROOT_PATH):
		self.data_path = os.path.join(ROOT_PATH, "musan","wav")
		self.save_path = os.path.join(dir, "custom","wav")
		self.pickle_path = os.path.join(ROOT_PATH, "musan","opensmile")
		self.noise_pickle_path = os.path.join(dir, "custom","opensmile")
		self.opensmile_manager = CREMASmileMaker(
			target_dir=self.save_path,
			save_dir=os.path.join(dir, "custom","opensmile")
		)
		self.noise_dataset = self.load_noises()

	def load_noises(self):
		pickle_filename = os.path.join(self.pickle_path,"emobase2010.pickle")
		if os.path.isfile(pickle_filename) == True:
			smile_maker = SmileMaker(save_dir=os.path.join(ROOT_PATH, "musan","opensmile"),
						target_dir=os.path.join(DATASET_PATH, "musan","noise"))
			# smile_maker.make_smile_csv()
			smile_maker.make_pickle_file()
		with open(pickle_filename, 'rb') as handle:
			data = pickle.load(handle)
			x_data = data['x_data']
			filenames = data['file_name']
			x_data = np.array(x_data)
			feat_mu = np.mean(x_data,axis=0)
			feat_st = np.std(x_data, axis=0)

			x_data  = normalization_ops(feat_mu, feat_st, x_data)
			return {'x_data':x_data,'file_name':filenames}

	def load_combined_voice(self):
		pickle_filename = os.path.join(self.noise_pickle_path,"emobase2010.pickle")
		with open(pickle_filename, 'rb') as handle:
			data = pickle.load(handle)
			x_data = data['x_data']
			y_data = data['y_data']
			filenames = data['file_name']
			x_data = np.array(x_data)
			feat_mu = np.mean(x_data,axis=0)
			feat_st = np.std(x_data, axis=0)

			x_data  = normalization_ops(feat_mu, feat_st, x_data)
			return {'x_data':x_data,'file_name':filenames, 'y_data':y_data}

	def combine_noise(self,voice_file:np.ndarray,noise_file,path=None,gain_during_overlay=10):
		voice = AudioSegment.from_file(voice_file)
		noise = AudioSegment.from_file(noise_file)
		combined:AudioSegment = voice.overlay(noise,gain_during_overlay=gain_during_overlay)

		voice_filename = os.path.basename(voice_file).split('.')[0]
		noise_filename = os.path.basename(noise_file).split('.')[0]

		save_path = os.path.join(self.save_path,f"{voice_filename}_{noise_filename}.wav")
		if path != None:
			save_path = os.path.join(path,f"{voice_filename}_{noise_filename}.wav")
		combined.export(save_path, format='wav')
		return save_path

	def get_max_diff_noise(self,voice_smile,feature_indexes):
		max_diff = 0
		max_diff_filename = ''
		base_features = voice_smile[feature_indexes]
		x_data = self.noise_dataset['x_data']
		filenames = self.noise_dataset['file_name']
		for data,filename in zip(x_data,filenames):
			compare_features = data[feature_indexes]
			diff:np.ndarray = np.abs(base_features-compare_features)
			diff:float = diff.sum()
			if max_diff < diff:
				max_diff = diff
				max_diff_filename = filename

		return max_diff_filename

	def extract_opensmile_csv(self):
		self.opensmile_manager.make_smile_csv()

	def get_opensmile_data(self,csv_path):
		return self.opensmile_manager.parse_smile_csv(csv_path)

	def make_pickle_file(self):
		self.opensmile_manager.make_pickle_file()