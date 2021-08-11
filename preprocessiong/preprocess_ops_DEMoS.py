# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:23:38 2021

@author: Youngdo Ahn
"""

from utils.setting import ROOT_PATH
import os
import numpy as np
import sys
import csv
sys.path.append("..")
from utils.functions import makedirs


AUDIO_DIR = "DEMoS_Italian_emotional_speech_corpus"
DB_DIR = "DEMoS"
PK_DIR = "emobase2010"  # IS09_emotion ComParE_2016 emobase2010
ORIGINAL_DATASETS_PATH = os.path.join(ROOT_PATH, AUDIO_DIR)
FEATURED_DATASETS_PATH = os.path.join(ROOT_PATH, DB_DIR)

emo_list = ['N', 'H', 'S', 'A']

def extract_lowlevel_features(audio_path):
    dir_OPS  = "C:/Users/USER/study/opensmile-3.0-win-x64"
    dir_conf = dir_OPS+'/config/emobase/'+PK_DIR+'.conf'
    #dir_conf = '/home/mspl/Young/toolkit/opensmile-2.3.0/config/'+PK_DIR+'.conf'

    # audio_path = '/home/mspl/Young/191027_multi/neu002.wav'
    dir_audi = audio_path
    dir_save = PK_DIR+'1.csv'

    cmd = dir_OPS+'/bin/SMILExtract -C '+dir_conf+' -I '+dir_audi+' -O '+dir_save    
    #cmd = 'SMILExtract -C '+dir_conf+' -I '+dir_audi+' -O '+dir_save
    # 혹시 원하시면 프레임별로 lld feature 추출도 가능합니다. Opensmile 명령어 입력하실 때 아웃풋 파일 이름 앞에 옵션으로 -D 넣으시면 돼요.
    # 오픈스마일 가이드북 35페이지에 나와있습니다.
    os.system(cmd)          
    csvfile = open(dir_save,'r')
    reader = [each for each in csv.DictReader(csvfile, delimiter=';')]
    csvfile.close()
    feats=str(reader[-1]).split(',')[1:-1]#[2:-1]
    cmd = 'rm '+dir_save
    os.system(cmd) 
    #cmd = 'rm smile.log'
    #os.system(cmd) 
    return feats


x_data = []
y_data = []
g_data = []
s_data = []
makedirs(FEATURED_DATASETS_PATH)

n_list = np.sort(os.listdir(ORIGINAL_DATASETS_PATH+'/NEU/'))
for f_wav in n_list:
    f_info = f_wav.split('_')
    #x_data.append(extract_lowlevel_features(ORIGINAL_DATASETS_PATH+'/NEU/'+f_wav))
    y_data.append(0)
    if f_info[0] == 'm':
        g_data.append(0)
    else:
        g_data.append(1)
    s_data.append(int(f_info[1]))

e_list = np.sort(os.listdir(ORIGINAL_DATASETS_PATH+'/DEMOS/'))
for f_wav in e_list:
    f_info = f_wav.split('_')
    if f_info[-1][:3] == 'gio':
        y_data.append(1)
    elif f_info[-1][:3] == 'tri':
        y_data.append(2)
    elif f_info[-1][:3] == 'rab':
        y_data.append(3)
    else:
        continue
    #x_data.append(extract_lowlevel_features(ORIGINAL_DATASETS_PATH+'/DEMOS/'+f_wav))
    
    if f_info[1] == 'm':
        g_data.append(0)
    else:
        g_data.append(1)
    s_data.append(int(f_info[2]))
print('hi')
x_data = np.array(x_data)
y_data = np.array(y_data)
g_data = np.array(g_data)
s_data = np.array(s_data)

data = {'x_data':x_data, 'y_data':y_data, 'g_data':g_data, 's_data':s_data}
filename = "%s/%s.pickle" %(FEATURED_DATASETS_PATH, PK_DIR)
#with open(filename, 'wb') as handle:
#    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    
##
for en, emo in enumerate (emo_list):
    print('#emotion %s: %d' %(emo, sum(y_data==en)))
for gen in range(2):
    print('#gender %s: %d' %(gen, sum(g_data==gen)))
# spk 35 null 1~69,, avg 70
for spk in range(np.min(s_data),np.max(s_data)+1):
    print('#speaker %s: %d' %(spk, sum(s_data==spk)))