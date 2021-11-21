import abc
import os
import re
class DataReader:
    def __init__(self, data_path):
        self.data_path = data_path

    def get_file_list(self):
        file_list = []
        for root, _, files in os.walk(self.data_path):
            for audio_file in files:
                audio_file:str = audio_file

                if audio_file.split('.')[-1] !='wav' \
                or audio_file[0] == '.' \
                or self.is_valid_file(audio_file) == False:
                    continue
                file_list.append(os.path.join(root,audio_file))
        return file_list

    @abc.abstractclassmethod
    def is_valid_file(self,filename):
        pass 

class CremaReader(DataReader):
    def is_valid_file(self,filename):
        if filename.split('_')[2] in ['NEU','HAP','SAD','ANG']:
            return True
        return False

class MusanReader(DataReader):
    def is_valid_file(self,filename):
        return True

class IemocapReader(DataReader):
    def load_wav_emotion_dict(self):
        self.wav_emotion_dict = {}
        for session_num in range(1,6):
            emotion_file_path = os.path.join(self.input_dir,f"Session{session_num}","dialog","EmoEvaluation")
            for root, _, files in os.walk(emotion_file_path):
                for file in files:
                    if file[0] == '.' or file.split('.')[-1] != "txt":
                        continue
                    with open(os.path.join(root,file),'r') as f:
                        lines = f.readlines()
                        if "% [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]" not in lines[0]:
                            continue
                        for line in lines:
                            pattern = re.compile("\[[0-9.]* - [0-9.]*\]	Ses[0-5][0-5][M,F]_.*	[a-z]*	\[.*\]")
                            if pattern.match(line) == None:
                                continue
                            infos = line.split('\t')
                            filename = infos[1]
                            emotion = infos[2].upper()
                            if emotion == 'EXC':
                                emotion = 'HAP'
                            if emotion not in ['NEU','HAP','SAD','ANG','EXC']:
                                continue
                            self.wav_emotion_dict[f"{filename}.wav"] = emotion

    def get_label(self,filename):
        if self.wav_emotion_dict == None:
            self.load_wav_emotion_dict()
        if filename not in self.wav_emotion_dict.keys():
            return None
        return self.wav_emotion_dict[filename]

    def is_valid_file(self,filename):
        return self.get_label(filename) != None
