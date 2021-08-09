from setting import ROOT_PATH
from pydub import AudioSegment
import numpy as np
import os
from .opensmile_maker import SmileMaker

class Noise_Combiner():
	def __init__(self):
		self.noise_dataset = []
		self.save_path = os.path.join(ROOT_PATH, "custom","wav")
		self.opensmile_manager = SmileMaker(
			target_dir=self.save_path,
			save_dir=os.path.join(ROOT_PATH, "custom","opensmile")
		)

	def combine_noise(self,voice_file,noise_file,gain_during_overlay=None):
		sound1 = AudioSegment.from_file(voice_file)
		sound2 = AudioSegment.from_file(noise_file)
		combined:AudioSegment = sound1.overlay(sound2,gain_during_overlay=gain_during_overlay)

		voice_filename = os.path.basename(voice_file).split('.')[0]
		noise_filename = os.path.basename(noise_file).split('.')[0]
		combined.export(os.path.join(self.save_path,f"{voice_filename}_{noise_filename}.wav"), format='wav')

	def get_max_diff_noise(self,voice_smile,feature_indexes):
		max_diff = 0
		max_diff_filename = ''
		base_features = voice_smile[feature_indexes]
		x_data = self.noise_dataset['x_data']
		for i,data in enumerate(x_data):
			compare_features = data[feature_indexes]
			diff:np.ndarray = np.abs(base_features-compare_features)
			diff:float = diff.sum()
			if max_diff < diff:
				max_diff = diff
				max_diff_filename = self.noise_dataset['y_data'][i]

		return max_diff_filename

	def extract_opensmile_csv(self,path):
		self.opensmile_manager.make_smile_csv()

	def get_opensmile_data(self,csv_path):
		return self.opensmile_manager.parse_smile_csv(csv_path)