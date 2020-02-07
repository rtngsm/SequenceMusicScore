# -*- coding: utf-8 -*-
"""
Created on 2019/9/22
run the main result.
@author: rtn
"""
if __name__=="__main__":
    from tqdm import tqdm
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imp import reload
    import os
    import copy
    #os.chdir("C:\\Users\\xiaoy\\Desktop\\modify\\code")
    import estimator as es
    reload(es)
    
    total = pd.read_csv("totaldata_newscore_withnum.csv",sep = ",",encoding = 'gbk') 
    data_transition = pd.read_csv("data_transition.csv",sep = ",",encoding = 'gbk') 
    bic = pd.read_csv("BIC_new.csv")
    bic_minus = set(bic['pair'][bic['diff']>0])
    pitch , note = es.pitch_note()    
    cat = list(set(total['type'])) 
    col = ["NB","DT","SVM","Network","SNB+FULL","SNB+BIC","SVM+Audio","LR_SVM","DT_Audio","LR_DT","Network_Audio","LR_network","NB_Audio","LR_NB"]
    accuracy = pd.DataFrame(columns = col)    
    #################################################### main loop
    for r in tqdm(range(0,1000)):
        # train and test 
        trainset , testset , train_transition, test_transition = train_test_split(total,data_transition,test_size = 0.5,random_state=r)
        # estimate the parameter
        pi_before,p_first_pitch,p_first_beat,p_transfer = es.estimator(trainset)
        # Bic selection
        p_transfer_bic = copy.deepcopy(p_transfer)
        p_transfer_bic = es.BIC_estimate(bic_minus,p_transfer_bic,cat)
        posterior_test,accu_snb = es.Posterior(testset,pi_before,p_first_pitch,p_first_beat,p_transfer,cat,pitch,note)
        posterior_bic_test,accu_snb_bic = es.Posterior(testset,pi_before,p_first_pitch,p_first_beat,p_transfer_bic,cat,pitch,note)
        # combination
        posterior_bic_train, accu_snb_bic_train = es.Posterior(trainset,pi_before,p_first_pitch,p_first_beat,p_transfer_bic,cat,pitch,note)
        # SVM、DT、Network based on audio feature
        posterior_svm_train, posterior_svm_test, accu_svm_audio = es.SVM_AUDIO(trainset,testset,cat)
        posterior_dt_train, posterior_dt_test, accu_dt_audio = es.DT_AUDIO(trainset,testset,cat)
        posterior_network_train, posterior_network_test, accu_network_audio = es.NETWORK_AUDIO(trainset,testset,cat)
        posterior_nb_train, posterior_nb_test, accu_nb_audio = es.NB_AUDIO(trainset,testset,cat)
	# combination
        accu_LR_with_svm = es.LR(posterior_bic_train,posterior_svm_train,trainset['music_type'],posterior_bic_test,posterior_svm_test,testset['music_type'])
        accu_LR_with_dt = es.LR(posterior_bic_train,posterior_dt_train,trainset['music_type'],posterior_bic_test,posterior_dt_test,testset['music_type'])
        accu_LR_with_network = es.LR(posterior_bic_train,posterior_network_train,trainset['music_type'],posterior_bic_test,posterior_network_test,testset['music_type'])
        accu_LR_with_nb = es.LR(posterior_bic_train,posterior_nb_train,trainset['music_type'],posterior_bic_test,posterior_nb_test,testset['music_type'])
	# SVM、DT、Network,NB based on marginal frequency
        accu_dt = es.dt(train_transition,test_transition)
        accu_svm = es.svm_tran(train_transition,test_transition)
        accu_net = es.network(train_transition,test_transition)
        accu_nb = es.nb(train_transition,test_transition)
        
        accuracy.loc[r] = [accu_nb,accu_dt,accu_svm,accu_net,accu_snb,accu_snb_bic,accu_svm_audio,accu_LR_with_svm,accu_dt_audio,accu_LR_with_dt,accu_network_audio,accu_LR_with_network,accu_nb_audio,accu_LR_with_nb]
    accuracy = pd.DataFrame(accuracy)    
    accuracy.to_csv("accuracy_all_0930.csv")
    result = accuracy.describe()
    result.to_csv("result_0930.csv")

