import re
from utils.functions import makedirs, snr
from pydub import AudioSegment
import numpy as np
import os
import random
from scipy.io.wavfile import read
from tqdm import tqdm

class NoisySoundGenerator():
    def __init__(self,voice_list,noise_list):
        self.voice_list = voice_list
        self.noise_list = noise_list

    def combine_noise(self,voice_file:str,noise_file:str,save_path:str,dB_diff:float):
        makedirs(save_path)
        voice_filename = re.split('[/\\\\.]',voice_file)[-2]# voice_file.split('/\\')[-1].split('.')[0]
        noise_filename = re.split('[/\\\\.]',noise_file)[-2]#noise_file.split('/\\')[-1].split('.')[0]
        file_path = os.path.join(save_path,f"{voice_filename}_{noise_filename}.wav")
        if os.path.exists(file_path):
            return

        voice = AudioSegment.from_file(voice_file)
        noise = AudioSegment.from_file(noise_file)
        combined:AudioSegment = voice.overlay(noise,gain_during_overlay=dB_diff)
        
        combined.export(file_path, format='wav')
    # def amplitude_noise_to_targe_snr(self,target:float,now:float,noise:np.ndarray):
    #     factor = 10^(now - target/20)
    #     return noise*factor

    def generate(self,save_path,dB):
        for voice_file in tqdm(self.voice_list):
            voice,channel = read(voice_file)
            voice:np.ndarray = np.array(voice,dtype=float)

            noise_file = random.choice(self.noise_list)
            noise,channel = read(noise_file)
            noise:np.ndarray = np.array(noise,dtype=float)
            now_snr = snr(voice,noise)
            # noise = self.amplitude_noise_to_targe_snr(dB,now_snr)
            self.combine_noise(voice_file,noise_file,save_path,dB - now_snr)

    def remove_generated_files(self):
        for filename in os.listdir(self.wav_path):
            file_path = os.path.join(self.wav_path, filename)
            os.unlink(file_path)
        for filename in os.listdir(self.smile_dir):
            file_path = os.path.join(self.smile_dir, filename)
            if file_path.split('.')[-1] == 'csv':
                os.unlink(file_path)