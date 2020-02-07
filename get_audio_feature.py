#!usr/bin/env.python
# _*_ coding: utf-8 _*_
import os
import librosa
import json
import numpy as np
import pandas as pd
import mido
from mido import MidiFile
import midi2audio
from midi2audio import FluidSynth
np.set_printoptions(threshold=1e6)

col=['music_name','file_type','music_type','music_duration','mean_ZeroCrossingRate','std_ZeroCrossingRate','mean_RootMeanSquareEnergy',
     'std_RootMeanSquareEnergy','mean_spectral_centroid','std_spectral_centroid','mean_spectral_flatness','sdt_spectral_flatness',
    'mean_spectral_roll_off','std_spectral_roll_off' ]
for i in range(0,6):
    col.append('mean_Tonnetz_dimension_'+str(i))
    col.append('std_Tonnetz_dimension_'+str(i))
for i in range(0,13):
    col.append('mean_MFCC_'+str(i))      
    col.append('std_MFCC_'+str(i)) 
data = pd.DataFrame(columns = col)
index = 0

filepath_1 = "C:/Users/admin/Desktop/MP3_data/mp3_classic/"
filename_1 = os.listdir(filepath_1)
filepath_2 = "C:/Users/admin/Desktop/MP3_data/mp3_folk/"
filename_2 = os.listdir(filepath_2)
filepath_3 = "C:/Users/admin/Desktop/MP3_data/mp3_jazz/"
filename_3 = os.listdir(filepath_3)

for file in filename_1:
    print(file)
    music,sr = librosa.load(filepath_1+"/"+file,sr = None)
    new = []
    new.append(file.split(".")[0]) ### music_name
    new.append("mp3")    ###file_type
    new.append("classic")  ###music_type
    new.append(librosa.get_duration(music,sr = sr)) #音乐时间
    zcr = librosa.feature.zero_crossing_rate(music)
    new.append(np.mean(zcr))  #过零率
    new.append(np.std(zcr))
    rmse = librosa.feature.rmse(music)
    new.append(np.mean(rmse)) #平均帧能量
    new.append(np.std(rmse))
    spec = librosa.feature.spectral_centroid(music,sr = sr)  ##光谱矩心
    new.append(np.mean(spec))
    new.append(np.std(spec))
    flatness = librosa.feature.spectral_flatness(music)
    new.append(np.mean(flatness))
    new.append(np.std(flatness))
    roll_off = librosa.feature.spectral_rolloff(music,sr =sr)
    new.append(np.mean(roll_off))
    new.append(np.std(roll_off))
    
    Tonn = librosa.feature.tonnetz(music,sr = sr)  ##音调质心，6个维度，0-5
    for i in range(0,6):
        new.append(np.mean(Tonn[i,]))
        new.append(np.std(Tonn[i,]))
    mfcc = librosa.feature.mfcc(music,sr = sr,n_mfcc=13)
    for i in range(0,13):
        new.append(np.mean(mfcc[i,]))
        new.append(np.std(mfcc[i,]))        
    data.loc[index] = new
    index = index +1

for file in filename_2:
    print(file)
    music,sr = librosa.load(filepath_2+"/"+file,sr = None)
    new = []
    new.append(file.split(".")[0]) ### music_name
    new.append("mp3")    ###file_type
    new.append("folk")  ###music_type
    new.append(librosa.get_duration(music,sr = sr)) 
    zcr = librosa.feature.zero_crossing_rate(music)
    new.append(np.mean(zcr))  
    new.append(np.std(zcr))
    rmse = librosa.feature.rmse(music)
    new.append(np.mean(rmse)) 
    new.append(np.std(rmse))
    spec = librosa.feature.spectral_centroid(music,sr = sr)  
    new.append(np.mean(spec))
    new.append(np.std(spec))
    flatness = librosa.feature.spectral_flatness(music)
    new.append(np.mean(flatness))
    new.append(np.std(flatness))
    roll_off = librosa.feature.spectral_rolloff(music,sr =sr)
    new.append(np.mean(roll_off))
    new.append(np.std(roll_off))
    
    Tonn = librosa.feature.tonnetz(music,sr = sr)  
    for i in range(0,6):
        new.append(np.mean(Tonn[i,]))
        new.append(np.std(Tonn[i,]))
    mfcc = librosa.feature.mfcc(music,sr = sr,n_mfcc=13)
    for i in range(0,13):
        new.append(np.mean(mfcc[i,]))
        new.append(np.std(mfcc[i,]))        
    data.loc[index] = new
    index = index +1

for file in filename_3:
    print(file)
    music,sr = librosa.load(filepath_3+"/"+file,sr = None)
    new = []
    new.append(file.split(".")[0]) ### music_name
    new.append("mp3")    ###file_type
    new.append("jazz")  ###music_type
    new.append(librosa.get_duration(music,sr = sr)) 
    zcr = librosa.feature.zero_crossing_rate(music)
    new.append(np.mean(zcr))  
    new.append(np.std(zcr))
    rmse = librosa.feature.rmse(music)
    new.append(np.mean(rmse)) 
    new.append(np.std(rmse))
    spec = librosa.feature.spectral_centroid(music,sr = sr)  
    new.append(np.mean(spec))
    new.append(np.std(spec))
    flatness = librosa.feature.spectral_flatness(music)
    new.append(np.mean(flatness))
    new.append(np.std(flatness))
    roll_off = librosa.feature.spectral_rolloff(music,sr =sr)
    new.append(np.mean(roll_off))
    new.append(np.std(roll_off))
    
    Tonn = librosa.feature.tonnetz(music,sr = sr)  
    for i in range(0,6):
        new.append(np.mean(Tonn[i,]))
        new.append(np.std(Tonn[i,]))
    mfcc = librosa.feature.mfcc(music,sr = sr,n_mfcc=13)
    for i in range(0,13):
        new.append(np.mean(mfcc[i,]))
        new.append(np.std(mfcc[i,]))        
    data.loc[index] = new
    index = index +1
print(index)
data.to_csv("C:/Users/admin/Desktop/MP3_data/feature.csv",index = False)
