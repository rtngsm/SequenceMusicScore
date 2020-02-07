# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:36:32 2020
Use wav_feture data, KNN and LDC classifers to implement the voting method. 
@author: rtn
"""

import pandas as pd
import numpy as np
import os
import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


address = "D:\\MP3_data\\wav_data\\"
file = os.listdir(address)

data_22_16 = pd.read_csv(address+'wav_feature_22_16.csv')
data_22_32 = pd.read_csv(address+'wav_feature_22_32.csv')
data_44_16 = pd.read_csv(address+'wav_feature_44_16.csv')
data_44_32 = pd.read_csv(address+'wav_feature_44_32.csv')
title = data_22_16.columns


def KNN(trainset,testset):    
    knn = KNeighborsClassifier(n_neighbors=10)
    x_train = trainset[title[2:22]]
    
    y_train = trainset['genre']
    x_test = testset[title[2:22]]
    y_test = testset['genre']
    knn.fit(x_train,y_train)
    pred = knn.predict(x_test)
    pred_code = []
    for p in pred:
        if p=="classic":
            pred_code.append(np.array([1,0,0]))
        if p=="jazz":
            pred_code.append(np.array([0,1,0]))
        if p=="folk":
            pred_code.append(np.array([0,0,1]))
    score = knn.score(x_test, y_test)
    
    return pred,pred_code,score


def LDC(trainset,testset):    
    ldc = LinearDiscriminantAnalysis()
    x_train = trainset[title[2:22]]    
    y_train = trainset['genre']
    x_test = testset[title[2:22]]
    y_test = testset['genre']
    ldc.fit(x_train,y_train)
    pred = ldc.predict(x_test)
    pred_code = []
    for p in pred:
        if p=="classic":
            pred_code.append(np.array([1,0,0]))
        if p=="jazz":
            pred_code.append(np.array([0,1,0]))
        if p=="folk":
            pred_code.append(np.array([0,0,1]))
    score = ldc.score(x_test, y_test)
    
    return pred,pred_code,score

col = ["knn_22_16","knn_22_32","knn_44_16","knn_44_32","ldc_22_16","ldc_22_32","ldc_44_16","ldc_44_32","vote_const","vote_accuracy"]
table = pd.DataFrame(columns = col)
for r in range(1000):

    # train and test
    ####1:knn_22_16,2:knn_22_32,3:knn_44_16,4:knn_44_32,5:ldc_22_16,6:ldc_22_32,7:ldc_44_16,8:ldc_44_32
    train_22_16, test_22_16, train_22_32, test_22_32, train_44_16, test_44_16, train_44_32, test_44_32 = train_test_split(data_22_16, data_22_32,data_44_16, data_44_32, test_size=0.5, random_state = r)
    pred_1 , pred_1_code, score_1 = KNN(train_22_16,test_22_16) 
    pred_2 , pred_2_code, score_2 = KNN(train_22_32,test_22_32) 
    pred_3 , pred_3_code, score_3 = KNN(train_44_16,test_44_16) 
    pred_4 , pred_4_code, score_4 = KNN(train_44_32,test_44_32) 
    
    pred_5 , pred_5_code, score_5 = LDC(train_22_16,test_22_16) 
    pred_6 , pred_6_code, score_6 = LDC(train_22_32,test_22_32) 
    pred_7 , pred_7_code, score_7 = LDC(train_44_16,test_44_16) 
    pred_8 , pred_8_code, score_8 = LDC(train_44_32,test_44_32) 
    
    y_test = test_22_16['genre']
    #pred_table = {"p1":pred_1_code,"p2":pred_2_code,"p3":pred_3_code,"p4":pred_4_code,"p5":pred_5_code,"p6":pred_6_code,"p7":pred_7_code,"p8":pred_8_code}
    #pred_table = np.array(pred_table)
    
    w_const = np.array([1,1,1,1,2,2,2,2])  ###weight setting
    w_accuracy = np.array([score_1,score_2,score_3,score_4,score_5,score_6,score_7,score_8])
    
    ####classic:(1,0,0),jazz(0,1,0),folk(0,0,1)

    vote_code_const = np.array(pred_1_code)*w_const[0]+np.array(pred_2_code)*w_const[1]+\
                        np.array(pred_3_code)*w_const[2]+np.array(pred_4_code)*w_const[3]+\
                        np.array(pred_5_code)*w_const[4]+np.array(pred_6_code)*w_const[5]+\
                        np.array(pred_7_code)*w_const[6]+np.array(pred_8_code)*w_const[7]
    vote_code_accuracy = np.array(pred_1_code)*w_accuracy[0]+np.array(pred_2_code)*w_accuracy[1]+\
                        np.array(pred_3_code)*w_accuracy[2]+np.array(pred_4_code)*w_accuracy[3]+\
                        np.array(pred_5_code)*w_accuracy[4]+np.array(pred_6_code)*w_accuracy[5]+\
                        np.array(pred_7_code)*w_accuracy[6]+np.array(pred_8_code)*w_accuracy[7]
    
    ### accuracy after voting
    vote_code_const = np.apply_along_axis(np.argmax, 1, vote_code_const)
    vote_code_accuracy = np.apply_along_axis(np.argmax, 1, vote_code_accuracy)

    ###vote_code 0：classic，1：jazz，2：folk
    pred_vote_const = []
    pred_vote_accuracy = []

    for p in vote_code_const:
        if p==0:
            pred_vote_const.append("classic")
        if p==1:
            pred_vote_const.append("jazz")
        if p==2:
            pred_vote_const.append("folk")
    for p in vote_code_accuracy:
        if p==0:
            pred_vote_accuracy.append("classic")
        if p==1:
            pred_vote_accuracy.append("jazz")
        if p==2:
            pred_vote_accuracy.append("folk")
    
    score_vote_accuracy = sum(pred_vote_accuracy==y_test)/len(y_test)
    score_vote_const = sum(pred_vote_const==y_test)/len(y_test)
    
    ta = [score_1,score_2,score_3,score_4,score_5,score_6,score_7,score_8,score_vote_const,score_vote_accuracy]
    table.loc[r] = ta
    print(r)
    
table.mean(axis=0)
table.to_csv("D:\\MP3_data\\wav_data\\result.csv",index = "False")
        
