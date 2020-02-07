# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 09:25:55 2018

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 14 20:56:07 2018

@author: admin
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import seaborn as sns
import xml.dom.minidom as x
from xml.dom.minidom import parse,parseString
from collections import Counter
octave = [1,2,3,4,5,6,7]
step = ["C","C1","D-1","D","D1","E-1","E","F","F1","G-1","G","G1","A-1","A","A1","B-1","B"]
pitch = []
pitch.append("0A")
pitch.append("0A1")
pitch.append("0B-1")
pitch.append("0B")
for i in octave:
    for j in step:
        pitch.append(str(i)+j)
#pitch.remove('A7')
#pitch.remove('B7')
pitch.append('8C')
pitch.append('rest') ## 110 nodes, according to pinao keyboard

pitch_m = []
for i in pitch:
    if "-1" in i:
        continue
    else:
        pitch_m.append(i) ##only 89 nodes in pitch space
note = ["32nd","16th","eighth","quarter","half","whole"]

#### get score sequence from xml files
def get_score(content):
    Note = content.getElementsByTagName("note")
    #get the text from the Text Node within the <step>,
    #and convert it from unicode to ascii
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    t5 = []
    t6 = []
    for n in Note:
        try:
            step = n.getElementsByTagName("step")
            t1.append(str(step[0].childNodes[0].nodeValue))
        except:
            t1.append(str("rest"))
        try:
            octave = n.getElementsByTagName("octave")
            t2.append(str(octave[0].childNodes[0].nodeValue))
        except:
            t2.append(str(""))
        try:
            alter = n.getElementsByTagName("alter")
            t5.append(str(alter[0].childNodes[0].nodeValue))
        except:
            t5.append(str(""))
        try:
            duration = n.getElementsByTagName("duration")
            t3.append(str(duration[0].childNodes[0].nodeValue))
        except:
            t3.append(str(""))
        try:
            type_ = n. getElementsByTagName("type")
            t4.append(str(type_[0].childNodes[0].nodeValue))
        except:
            t4.append(str("rest"))  
    
        chord = n.getElementsByTagName("chord")
        if chord==[]:
            t6.append(0)
        else:
            t6.append(1)        
    Matri = pd.DataFrame(np.vstack((t1,t2,t3,t4,t5,t6)))
    for i in range(0,len(t1)):
        Matri.loc[6,i] = str(Matri.iloc[1,i])+str(Matri.iloc[0,i])+str(Matri.iloc[4,i])
    for i in range(0,len(t1)):
        if "-1" in Matri.loc[6][i]:
            Matri.loc[6][i] = pitch[pitch.index(Matri.loc[6][i])-1] 
    for i in range(0,len(t1)):  
        if Matri.loc[6][i] not in pitch_m:
           pitch_exist = Matri.loc[6][i][0:2]
           switch = 1 
           Matri.loc[6][i] = pitch_m[pitch_m.index(pitch_exist)+switch]
    for i in range(0,len(t2)):
        if Matri.loc[3][i]=='rest':
           del Matri[i] 
    Matri.columns = range(0,Matri.shape[1])
    return Matri




def get_frequency(content,musicname):
    score = get_score(content)
    #num_octave = [] 
    #num_step = []  
    feature = []
    for h in pitch_m:
        feature.append(list(score.loc[6]).count(h))
    length = sum(feature)
    for b in note:
        feature.append(list(score.loc[3]).count(h))
    feature.append(length)
    return feature
 
## calculate the data we use for SNB method
def get_matrix(content,musicname):
    matrix_pitch = pd.DataFrame(0,columns=pitch_m,index = pitch_m) 
    matrix_type = pd.DataFrame(0,columns = note,index = note)
    pitch_num = []
    beat_num = []
    score = get_score(content)
    
    length_effective = 0
    first_note = [score.loc[6,0],score.loc[3,0]]  
    for i in range(0,score.shape[1]-1):
        j = i+1
        pitch_now = score.loc[6,i]
        pitch_follow = score.loc[6,j]
  
        if pitch_now=="rest" and pitch_follow == "rest":
            continue
        else:            
            length_effective = length_effective+1
            matrix_pitch.ix[pitch_now,pitch_follow] = matrix_pitch.ix[pitch_now,pitch_follow]+1
            type_now = score.loc[3,i]
            type_follow = score.loc[3,j]
            matrix_type.ix[type_now,type_follow] = matrix_type.ix[type_now,type_follow]+1  
    for i in range(0,matrix_pitch.shape[1]):
        num = sum(matrix_pitch.iloc[i])
        pitch_num.append(num)
        
    for i in range(0,matrix_type.shape[1]):
        num = sum(matrix_type.iloc[i])
        beat_num.append(num)
   
    return matrix_pitch,matrix_type,length_effective,first_note,pitch_num,beat_num



filepath_1 = "D:\\xml\\jazz"
filename_1 = os.listdir(filepath_1)
filepath_2 = "D:\\xml\\folk"
filename_2 = os.listdir(filepath_2)
filepath_3 = "D:\\xml\\classic"
filename_3 = os.listdir(filepath_3)



col=[]
for i in range(0,89*89+6*6+1+2+1+1+89+6): # add Length, First pitch, First beat,genre,music name
   col.append(str(i))

data = pd.DataFrame(columns = col)

index = 0

for file in filename_1:
    try:
        content = x.parse(filepath_1+"\\"+str(file))
        music_name = file.split('.xml')[0]
       # score_tem = get_score(content)
        m_pitch, m_type,length_effective,first_note,pitch_num,beat_num = get_matrix_chord_2(content,music_name)
        m1 = m_pitch.values.reshape(89*89)
        m2 = m_type.values.reshape(6*6)
        m = np.hstack((m1,m2))
        m = list(m)
        m.append(length_effective)
        m.append(first_note[0])
        m.append(first_note[1])
        m.append("jazz")
        m.append(music_name)
        m = m + pitch_num
        m = m + beat_num
        data.loc[index] = m
        index = index+1
        print(file+str(1))
    except:
        print("failed"+file)
        continue

for file in filename_2:
    try:
        content = x.parse(filepath_2+"\\"+str(file))
        music_name = file.split('.xml')[0]
       # score_tem = get_score(content)
        m_pitch, m_type,length_effective,first_note,pitch_num,beat_num = get_matrix_chord_2(content,music_name)
        m1 = m_pitch.values.reshape(89*89)
        m2 = m_type.values.reshape(6*6)
        m = np.hstack((m1,m2))
        m = list(m)
        m.append(length_effective)
        m.append(first_note[0])
        m.append(first_note[1])
        m.append("folk")
        m.append(music_name)
        m = m + pitch_num
        m = m + beat_num
        data.loc[index] = m
        
        index = index+1
        print(file+str(2))
    except:
        print("failed"+file)
        continue

for file in filename_3:
    try:       
        content = x.parse(filepath_3+"\\"+str(file))
        music_name = file.split('.xml')[0]
       # score_tem = get_score(content)
        m_pitch, m_type,length_effective,first_note,pitch_num,beat_num = get_matrix_chord_2(content,music_name)
        m1 = m_pitch.values.reshape(89*89)
        m2 = m_type.values.reshape(6*6)
        m = np.hstack((m1,m2))
        m = list(m)
        m.append(length_effective)
        m.append(first_note[0])
        m.append(first_note[1])
        m.append("classic")
        m.append(music_name)
        m = m + pitch_num
        m = m + beat_num
        data.loc[index] = m
        index = index+1
        print(file+str(3))
    except:
        print("failed"+file)
        continue
    
data.to_csv("E:/data_newscore_withnum.csv",index = False)
### including transitions and marginal frequency




