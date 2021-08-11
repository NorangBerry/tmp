from utils.functions import normalization_ops
import pickle
from utils.setting import DATASET_PATH, ROOT_PATH
from pydub import AudioSegment
import numpy as np
import os
from .opensmile_maker import CREMASmileMaker, SmileMaker

class Noise_Combiner():
	def __init__(self,save_path):
		self.wav_path = os.path.join(save_path,"wav")
		self.smile_dir = os.path.join(save_path,"opensmile")
		self.noise_pickle_path = os.path.join(ROOT_PATH, "musan","opensmile")
		self.opensmile_manager = CREMASmileMaker(
			wav_dir=self.wav_path,
			smile_dir=self.smile_dir
		)
		self.noise_dataset = self.load_noises()

	def load_noises(self):
		pickle_filename = os.path.join(self.noise_pickle_path,"emobase2010.pickle")
		# musan데이터를 아직 안뽑은 경우에만 돌도록
		if os.path.isfile(pickle_filename) == False:
			smile_maker = SmileMaker(
				wav_dir=os.path.join(DATASET_PATH, "musan","noise"),
				smile_dir=os.path.join(ROOT_PATH, "musan","opensmile")
			)
			smile_maker.make_smile_csv()
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
		pickle_filename = os.path.join(self.opensmile_manager.get_smile_path(),"emobase2010.pickle")
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

	def combine_noise(self,voice_file:np.ndarray,noise_file,gain_during_overlay=10):
		voice = AudioSegment.from_file(voice_file)
		noise = AudioSegment.from_file(noise_file)
		combined:AudioSegment = voice.overlay(noise,gain_during_overlay=gain_during_overlay)

		voice_filename = os.path.basename(voice_file).split('.')[0]
		noise_filename = os.path.basename(noise_file).split('.')[0]

		save_path = os.path.join(self.wav_path,f"{voice_filename}_{noise_filename}.wav")
		combined.export(save_path, format='wav')

		return save_path

	def get_max_diff_noise(self,voice_smile,feature_indexes,feature_name):
		max_diff = 0
		max_diff_filename = ''
		base_features = voice_smile[feature_indexes]
		x_data = self.noise_dataset['x_data']
		filenames = self.noise_dataset['file_name']
		for data,filename in zip(x_data,filenames):
			compare_features = data[feature_indexes]
			diff:np.ndarray = np.abs(base_features-compare_features)
			diff:float = diff.sum()
			# if self.need_upadte(diff,max_diff,feature_name):
			# 	max_diff = diff
			# 	max_diff_filename = filename
			if max_diff < diff:
				max_diff = diff
				max_diff_filename = filename

		return max_diff_filename

	# def need_upadte(self,new_val,old_val,name):
	# 	return new_val > old_val
			

	def extract_opensmile_csv(self):
		self.opensmile_manager.make_smile_csv()

	def make_pickle_file(self):
		self.opensmile_manager.make_pickle_file()

	def remove_generated_files(self):
		for filename in os.listdir(self.wav_path):
			file_path = os.path.join(self.wav_path, filename)
			os.unlink(file_path)
		for filename in os.listdir(self.smile_dir):
			file_path = os.path.join(self.smile_dir, filename)
			if file_path.split('.')[-1] == 'csv':
				os.unlink(file_path)