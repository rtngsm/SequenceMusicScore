# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 20:30:41 2020

@author: rtn
"""

import os
from pydub import AudioSegment
import tqdm
import pandas as pd
import numpy as np

xml_classic_address = "D:\\xml\\classic"
xml_folk_address = "D:\\xml\\folk"
xml_jazz_address = "D:\\xml\\jazz"
midi_classic_address = "D:\\MP3_data\\classic"
midi_folk_address = "D:\\MP3_data\\folk"
midi_jazz_address = "D:\\MP3_data\\jazz"


wav_classic_address_22_16 = "D:\\MP3_data\\classic_wav_22_16\\"
wav_classic_address_44_16 = "D:\\MP3_data\\classic_wav_44_16\\"
wav_jazz_address_22_16 = "D:\\MP3_data\\jazz_wav_22_16\\"
wav_jazz_address_44_16 = "D:\\MP3_data\\jazz_wav_44_16\\"
wav_folk_address_22_16 = "D:\\MP3_data\\folk_wav_22_16\\"
wav_folk_address_44_16 = "D:\\MP3_data\\folk_wav_44_16\\"


wav_classic_address_22_32 = "D:\\MP3_data\\classic_wav_22_32\\"
wav_classic_address_44_32 = "D:\\MP3_data\\classic_wav_44_32\\"
wav_jazz_address_22_32 = "E:\\jazz_wav_22_32\\"
wav_jazz_address_44_32 = "E:\\jazz_wav_44_32\\"
wav_folk_address_22_32 = "E:\\folk_wav_22_32\\"
wav_folk_address_44_32 = "F:\\folk_wav_44_32\\"

#os.mkdir(wav_classic_address_44_32)

mp3_classic_address = "D:\\MP3_data\\classic_mp3\\"
mp3_jazz_address = "D:\\MP3_data\\jazz_mp3\\"
mp3_folk_address = "D:\\MP3_data\\folk_mp3\\"

### mp3 2 wav, 22050、44100采样率的wav文件，16bit，8bit的文件无法生成
mp3_classic_list = os.listdir(mp3_classic_address)
for m in mp3_classic_list:
    sound = AudioSegment.from_mp3(mp3_jazz_address+m)
    name = m[:-4]
    sound2 = sound.set_frame_rate(22050) ##采样率
    sound2 = sound2.set_channels(1)
    sound2 = sound2.set_sample_width(4)  ###2等于16bit
    sound2.export(wav_classic_address_22_32+name+".wav", format="wav")
    sound3 = sound2.set_frame_rate(44100)
    sound3.export(wav_jazz_address_44_32+name+".wav", format="wav")
    print(name)


############################################提取特征，形成csv文件
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt


wav_classic_list = os.listdir(wav_classic_address_44_32)
wav_jazz_list = os.listdir(wav_jazz_address_44_32)
wav_folk_list = os.listdir(wav_folk_address_44_32)

col = ["name","genre","lowenergy",
       "zerocrossing_mean","centroid_mean","flux_mean","rolloff_mean",
       "MFCC_1_mean","MFCC_2_mean","MFCC_3_mean","MFCC_4_mean","MFCC_5_mean",
       "zerocrossing_var","centroid_var","flux_var","rolloff_var",
       "MFCC_1_var","MFCC_2_var","MFCC_3_var","MFCC_4_var","MFCC_5_var"]
table = pd.DataFrame(columns = col)
index = 0
for w in wav_classic_list:
    name = w[:-4]
    genre = "classic"
    [Fs, x] = audioBasicIO.readAudioFile(wav_classic_address_44_32+w)
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
    mean = F[0].mean(axis = 1)[[0,3,6,7,8,9,10,11,12]]
    var = F[0].var(axis = 1)[[0,3,6,7,8,9,10,11,12]]
    ###low energy
    lowenergy = sum(F[0][1]<F[0][1].mean())/len(F[0][1])
    
    ta = [name,genre,lowenergy]
    ta = np.hstack((ta,mean,var))
    
    table.loc[index] = ta
    index = index + 1
    if index%10==0:
        print(index)
        
for w in wav_jazz_list:
    name = w[:-4]
    genre = "jazz"
    [Fs, x] = audioBasicIO.readAudioFile(wav_jazz_address_44_32+w)
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
    mean = F[0].mean(axis = 1)[[0,3,6,7,8,9,10,11,12]]
    var = F[0].var(axis = 1)[[0,3,6,7,8,9,10,11,12]]
    ###low energy
    lowenergy = sum(F[0][1]<F[0][1].mean())/len(F[0][1])
    
    ta = [name,genre,lowenergy]
    ta = np.hstack((ta,mean,var))
    
    table.loc[index] = ta
    index = index + 1
    if index%10==0:
        print(index)
        
for w in wav_folk_list:
    name = w[:-4]
    genre = "folk"
    [Fs, x] = audioBasicIO.readAudioFile(wav_folk_address_44_32+w)
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
    mean = F[0].mean(axis = 1)[[0,3,6,7,8,9,10,11,12]]
    var = F[0].var(axis = 1)[[0,3,6,7,8,9,10,11,12]]
    ###low energy
    lowenergy = sum(F[0][1]<F[0][1].mean())/len(F[0][1])
    
    ta = [name,genre,lowenergy]
    ta = np.hstack((ta,mean,var))
    
    table.loc[index] = ta
    index = index + 1
    if index%10==0:
        print(index)

table.to_csv("D:\MP3_data\wav_data\\wav_feature_44_32.csv")

[Fs, x] = audioBasicIO.readAudioFile("D:/MP3_data/classic_wav/01 Menuet22_8.wav")
[Fs, x] = audioBasicIO.readAudioFile("D:\\MP3_data\\classic_wav_44_16\\01. Kyrie Eleison.wav")
Fs.close
x = x.astype(float)
xx = x.sum(axis = 1)/2
F = audioFeatureExtraction.stFeatureExtraction(xx, Fs, 0.050*Fs, 0.025*Fs);




