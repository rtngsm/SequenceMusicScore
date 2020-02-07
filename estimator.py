# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 10:41:01 2019
functions used in main.py
@author: rtn
"""

import numpy as np
import pandas as pd
import copy
from sklearn import svm,tree
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def pitch_note():
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
    pitch.append('8C')
    pitch.append('rest')     
    pitch_m = []
    for i in pitch:
        if "-1" in i:
            continue
        else:
            pitch_m.append(i) ##89 nodes according to piano keyboard
    note = ["32nd","16th","eighth","quarter","half","whole"]
    return pitch_m,note



def estimator(trainset): 
    cat = list(set(trainset['type'])) # type of music 
    pitch , note = pitch_note()
    n = trainset.shape[0]
    N = {}
    pi_before = {}
    p_transfer = {}
    p_first_pitch = {}
    p_first_beat = {}
    for c in cat:
            N[c]=0
            pi_before[c]=0
            p_transfer[c] = np.array([0]*7957)
            p_first_pitch[c] = np.array([0]*89)
            p_first_beat[c] = np.array([0]*6)
    for i in range(0,trainset.shape[0]):
        #print(i)
        ##estimate the prior probability
        for c in cat:
            if list(trainset['type'])[i] ==c:
                N[c] = N[c]+1
                p_transfer[c] = p_transfer[c]+np.array(trainset.iloc[i][1:7958])
            else:
                continue
        for c in cat:
            num = pitch.index(list(trainset['first_pitch'])[i])
            p_first_pitch[c][num] = p_first_pitch[c][num] + 1
            num_1 = note.index(list(trainset['first_beat'])[i])
            p_first_beat[c][num_1] = p_first_beat[c][num_1] + 1
    for c in cat:
        pi_before[c] = N[c]/n
        ######calculate the transitional probablities in pitch and beat
        for j in range(0,89):
            sum_p = sum(p_transfer[c][(j*89):(j*89+89)])
            if sum_p!=0:
                p_transfer[c][(j*89):(j*89+89)] = p_transfer[c][(j*89):(j*89+89)]/sum(p_transfer[c][(j*89):(j*89+89)])
            else:
                continue
        for j in range(0,6):
            sum_b = sum(p_transfer[c][(7921+j*6):(7921+j*6+6)])
            if sum_b!=0:
                p_transfer[c][(7921+j*6):(7921+j*6+6)] = p_transfer[c][(7921+j*6):(7921+j*6+6)]/sum(p_transfer[c][(7921+j*6):(7921+j*6+6)])
            else:
                continue
        #p_transfer[c] = p_transfer[c]/N[c]
        p_first_pitch[c] = p_first_pitch[c]/N[c]
        p_first_beat[c] = p_first_beat[c]/N[c]
 
        
    #plus 1e-3 in transional matrix
    for c in cat:
        for t in range(0,7957):
            if p_transfer[c][t]==0:
                p_transfer[c][t] = np.log(p_transfer[c][t] + 1e-3)
            else:
                p_transfer[c][t] = np.log(p_transfer[c][t])
        for t in range(0,89):
            if p_first_pitch[c][t]==0:
                p_first_pitch[c][t] = np.log(p_first_pitch[c][t]+1e-3)
            else:
                p_first_pitch[c][t] = np.log(p_first_pitch[c][t])
        for t in range(0,6):
            if p_first_beat[c][t]==0:
                p_first_beat[c][t] = np.log(p_first_beat[c][t]+1e-2)
            else:
                p_first_beat[c][t] = np.log(p_first_beat[c][t])
    return [pi_before,p_first_pitch,p_first_beat,p_transfer]

def BIC_estimate(bic_minus,p_transfer_bic,cat):
    for c in cat:
        for t in bic_minus:
            p_transfer_bic[c][t] = 0
    return p_transfer_bic
    
def Posterior(dataset,pi_before,p_first_pitch,p_first_beat,p_transfer,cat,pitch,note):
    posterior_log = pd.DataFrame(columns = cat)
    posterior_now = {}
    type_predicted = []
    for i in range(0,dataset.shape[0]):
        for c in cat:
            posterior_now[c] = np.log(pi_before[c]) + p_first_pitch[c][pitch.index(list(dataset['first_pitch'])[i])] + p_first_beat[c][note.index(list(dataset['first_beat'])[i])]
            posterior_now[c] = posterior_now[c]+ sum(dataset.iloc[i][1:7958]*p_transfer[c])
        type_predicted.append(max(posterior_now,key = posterior_now.get)) #
        posterior_log.loc[i] = posterior_now
    accuracy = get_accuracy(dataset,type_predicted)
    ####calculate the posterior probability 
    posterior = pd.DataFrame(columns = cat)
    posterior['jazz'] = 1/(1 + np.exp(posterior_log['folk']-posterior_log['jazz']) + np.exp(posterior_log['classic']-posterior_log['jazz']))
    posterior['folk'] = 1/(1 + np.exp(posterior_log['jazz']-posterior_log['folk']) + np.exp(posterior_log['classic']-posterior_log['folk']))
    posterior['classic'] = 1/(1 + np.exp(posterior_log['folk']-posterior_log['classic']) + np.exp(posterior_log['jazz']-posterior_log['classic']))
   
    return posterior, accuracy

def SVM_AUDIO(trainset,testset,cat):
    title = trainset.columns.values.tolist()
    x_train = trainset[title[8059:8107]]
    x_test = testset[title[8059:8107]]
    model = svm.SVC(kernel='poly',gamma=1.0/x_train.shape[1],probability=True)
    model.fit(x_train, trainset['music_type'])
    prob_train_svm_tmp = np.array(model.predict_proba(x_train))
    prob_test_svm_tmp = np.array(model.predict_proba(x_test))
    pred = model.predict(x_test)
    accu_svm_audio = get_accuracy(testset,pred)
    
    ####svm,the first column refers to the classic，the second refers to the folk，the third refers to the jazz
    posterior_svm_train = pd.DataFrame(columns = cat)
    posterior_svm_train['classic'] = prob_train_svm_tmp[:,0]
    posterior_svm_train['folk'] = prob_train_svm_tmp[:,1]
    posterior_svm_train['jazz'] = prob_train_svm_tmp[:,2]
    posterior_svm_train.columns = ['jazz_svm','folk_svm','classic_svm']
    
    posterior_svm_test = pd.DataFrame(columns = cat)
    posterior_svm_test['classic'] = prob_test_svm_tmp[:,0]
    posterior_svm_test['folk'] = prob_test_svm_tmp[:,1]
    posterior_svm_test['jazz'] = prob_test_svm_tmp[:,2]
    posterior_svm_test.columns = ['jazz_svm','folk_svm','classic_svm']
    
    return posterior_svm_train, posterior_svm_test, accu_svm_audio

def DT_AUDIO(trainset,testset,cat):
    title = trainset.columns.values.tolist()
    x_train = trainset[title[8059:8107]]
    x_test = testset[title[8059:8107]]
    model = tree.DecisionTreeClassifier(criterion = 'entropy')
    model.fit(x_train, trainset['music_type'])
    prob_train_dt_tmp = np.array(model.predict_proba(x_train))
    prob_test_dt_tmp = np.array(model.predict_proba(x_test))
    pred = model.predict(x_test)
    accu_dt_audio = get_accuracy(testset,pred)
    
    ####dt,the first column refers to the classic，the second refers to the folk，the third refers to the jazz
    posterior_dt_train = pd.DataFrame(columns = cat)
    posterior_dt_train['classic'] = prob_train_dt_tmp[:,0]
    posterior_dt_train['folk'] = prob_train_dt_tmp[:,1]
    posterior_dt_train['jazz'] = prob_train_dt_tmp[:,2]
    posterior_dt_train.columns = ['jazz_dt','folk_dt','classic_dt']
    
    posterior_dt_test = pd.DataFrame(columns = cat)
    posterior_dt_test['classic'] = prob_test_dt_tmp[:,0]
    posterior_dt_test['folk'] = prob_test_dt_tmp[:,1]
    posterior_dt_test['jazz'] = prob_test_dt_tmp[:,2]
    posterior_dt_test.columns = ['jazz_dt','folk_dt','classic_dt']
    
    return posterior_dt_train, posterior_dt_test, accu_dt_audio
    
def NETWORK_AUDIO(trainset,testset,cat):
    title = trainset.columns.values.tolist()
    x_train = trainset[title[8059:8107]]
    x_test = testset[title[8059:8107]]
    model = MLPClassifier(hidden_layer_sizes=(20,20,10),max_iter=500)
    model.fit(x_train, trainset['music_type'])
    prob_train_network_tmp = np.array(model.predict_proba(x_train))
    prob_test_network_tmp = np.array(model.predict_proba(x_test))
    pred = model.predict(x_test)
    accu_network_audio = get_accuracy(testset,pred)
    
    ####network,the first column refers to the classic，the second refers to the folk，the third refers to the jazz
    posterior_network_train = pd.DataFrame(columns = cat)
    posterior_network_train['classic'] = prob_train_network_tmp[:,0]
    posterior_network_train['folk'] = prob_train_network_tmp[:,1]
    posterior_network_train['jazz'] = prob_train_network_tmp[:,2]
    posterior_network_train.columns = ['jazz_network','folk_network','classic_network']
    
    posterior_network_test = pd.DataFrame(columns = cat)
    posterior_network_test['classic'] = prob_test_network_tmp[:,0]
    posterior_network_test['folk'] = prob_test_network_tmp[:,1]
    posterior_network_test['jazz'] = prob_test_network_tmp[:,2]
    posterior_network_test.columns = ['jazz_network','folk_network','classic_network']
    
    return posterior_network_train, posterior_network_test, accu_network_audio

def NB_AUDIO(trainset,testset,cat):
    title = trainset.columns.values.tolist()
    x_train = trainset[title[8059:8107]]
    x_test = testset[title[8059:8107]]
    model = GaussianNB() 
    model.fit(x_train, trainset['music_type'])
    prob_train_nb_tmp = np.array(model.predict_proba(x_train))
    prob_test_nb_tmp = np.array(model.predict_proba(x_test))
    pred = model.predict(x_test)
    accu_nb_audio = get_accuracy(testset,pred)
    
    ####nb,the first column refers to the classic，the second refers to the folk，the third refers to the jazz
    posterior_nb_train = pd.DataFrame(columns = cat)
    posterior_nb_train['classic'] = prob_train_nb_tmp[:,0]
    posterior_nb_train['folk'] = prob_train_nb_tmp[:,1]
    posterior_nb_train['jazz'] = prob_train_nb_tmp[:,2]
    posterior_nb_train.columns = ['jazz_nb','folk_nb','classic_nb']
    
    posterior_nb_test = pd.DataFrame(columns = cat)
    posterior_nb_test['classic'] = prob_test_nb_tmp[:,0]
    posterior_nb_test['folk'] = prob_test_nb_tmp[:,1]
    posterior_nb_test['jazz'] = prob_test_nb_tmp[:,2]
    posterior_nb_test.columns = ['jazz_nb','folk_nb','classic_nb']
    
    return posterior_nb_train, posterior_nb_test, accu_nb_audio
''' 
def LR(posterior_train_1,posterior_train_2,Train_y,posterior_test_1,posterior_test_2,Test_y):
    Train_x = posterior_train_1.join(posterior_train_2)
    Train_x = Train_x.ix[:,[0,1,3,4]] ##To avoid multicollinearity,select[["jazz","folk","jazz_svm","folk_svm"]]
    Test_x = posterior_test_1.join(posterior_test_2)
    Test_x = Test_x.ix[:,[0,1,3,4]] ##To avoid multicollinearity，select[["jazz","folk","jazz_svm","folk_svm"]]
    ############## Logitmodel
    lr = linear_model.LogisticRegression(multi_class='ovr')
    lr.fit(Train_x,Train_y)
    pred_lr = lr.predict(Test_x)
    accu_lr = sum(1*(pred_lr==Test_y))/len(Test_y) ###88.8%
    return accu_lr
'''

def get_keys(d, value):
    return [k for k,v in d.items() if v == value]
    
def get_accuracy(dataset,type_predicted):
    accuracy = sum(1*(type_predicted==dataset['music_type']))/dataset.shape[0] 
    return accuracy
    

def network(train_transition,test_transition):    
    x_train = train_transition.iloc[:,0:7957]
    x_test = test_transition.iloc[:,0:7957]
    y_train = train_transition.iloc[:,7958]
    y_test = test_transition.iloc[:,7958]
    model = MLPClassifier(hidden_layer_sizes=(20,20,10),max_iter=500)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    accuracy = sum(1*(pred==y_test))/len(y_test)
    return accuracy

def svm_tran(train_transition,test_transition):
    x_train = train_transition.iloc[:,0:7957]
    x_test = test_transition.iloc[:,0:7957]
    y_train = train_transition.iloc[:,7958]
    y_test = test_transition.iloc[:,7958]
    model = svm.SVC(kernel='poly',gamma=1.0/x_train.shape[1],probability=True)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    accuracy = sum(1*(pred==y_test))/len(y_test)
    return accuracy


def dt(train_transition,test_transition):
    x_train = train_transition.iloc[:,0:7957]
    x_test = test_transition.iloc[:,0:7957]
    y_train = train_transition.iloc[:,7958]
    y_test = test_transition.iloc[:,7958]
    model = tree.DecisionTreeClassifier(criterion = 'entropy')
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    accuracy = sum(1*(pred==y_test))/len(y_test)
    return accuracy

def nb(train_transition,test_transition):
    x_train = train_transition.iloc[:,0:7957]
    x_test = test_transition.iloc[:,0:7957]
    y_train = train_transition.iloc[:,7958]
    y_test = test_transition.iloc[:,7958]
    model = GaussianNB() 
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    accuracy = sum(1*(pred==y_test))/len(y_test)
    return accuracy
