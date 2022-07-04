#!/usr/bin/env python
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import torch
from torch.utils.data import *
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler 
import torch.optim as optim 

import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib 
from matplotlib import pyplot
import matplotlib.pyplot as plt 
import argparse
import os 
import shutil
import random
from sklearn.cluster import AgglomerativeClustering
import math

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.stats import chi2
import pandas as pd
import statsmodels.api as sm
import numpy as np 
import statsmodels.api as sm
from sklearn.metrics import auc, roc_auc_score, roc_curve

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn import metrics


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from DICE import yf_dataset_withdemo, model_2, collate_fn
from autoencoder_builder import get_auto_encoder, AutoEncoderEnum
from tqdm import tqdm
import wandb

def analysis_cluster_number_byclustering(data_cur, num_clusters, if_check, varname):
    data_C = data_cur.C
    data_v = data_cur.data_v
    data_y = data_cur.data_y

    list_c = data_C.tolist()
    list_onehot = []
    dict_c_count = {}
    dict_outcome_in_c_count = {}
    for i in range(num_clusters):
        dict_c_count[i] = 0 
        dict_outcome_in_c_count[i] = 0 
    
    for i in range(len(list_c)):
        temp = [0 for i in range(num_clusters)]
        temp[list_c[i]] = 1 
        list_onehot.append(temp)

        dict_c_count[list_c[i]] += 1 
        if data_y[i]==1:
            dict_outcome_in_c_count[list_c[i]] += 1 
    
    if if_check:
        print("--------")
        print("num_clusters=", num_clusters)
        print()
        print("list_c[0]=",list_c[0])
        print("list_onehot[0]=", list_onehot[0])
        print()
        print("list_c[1]=",list_c[1])
        print("list_onehot[1]=", list_onehot[1])
        print("--------")
    
    dict_outcome_ratio = {}
    for keyc in dict_c_count:
        if dict_c_count[keyc] == 0:
            dict_outcome_ratio[keyc] = 0
        else:
            dict_outcome_ratio[keyc] = dict_outcome_in_c_count[keyc]/dict_c_count[keyc]
    return dict_outcome_ratio, dict_c_count


def update_curset_pred_C_and_repD0420(args, model, data_cur, dataloader_cur, varname, datatrainM):
    print("-----------------")
    print("Deal with:", varname)
    #print("    update pred_C and pred_C")
    # update date_cur.rep
    final_embed = torch.randn(len(data_cur), args.n_hidden_fea, dtype=torch.float)
    model.eval()
    for batch_idx, (idx, data_x, data_v, target, batch_c) in enumerate(tqdm(dataloader_cur)):

        if args.cuda:
            data_x = data_x.cuda()
            data_v = data_v.cuda()
            target = target.cuda()

        encoded_x, decoded_x, output_c_no_activate, output_outcome = model(x=data_x, function="outcome_logistic_regression", demov=data_v)
        embed = encoded_x.data.cpu()
        for l in idx:
            final_embed[l.item()]=embed[l.item() % len(embed)]

    data_cur.rep = final_embed 
    representations = data_cur.rep
    for i in range(data_cur.rep.size()[0]):
        embed = representations[i,:]
        trans_embed = embed.view(embed.size()+(1,))
        xj = torch.norm(trans_embed - datatrainM.M, dim=0)
        new_cluster = torch.argmin(xj)
        data_cur.C[i] = new_cluster


def calculate_cluster_metrics(data_train):
    labels_pred = data_train.C.tolist()
    labels_true = data_train.data_y
    X = data_train.rep.numpy()
    print(type(X))
    score = {}
    score['silhouette_score'] = metrics.silhouette_score(X, labels_pred, metric='euclidean')
    score['calinski_harabasz_score'] = metrics.calinski_harabasz_score(X, labels_pred)
    score['davies_bouldin_score'] = metrics.davies_bouldin_score(X, labels_pred)
    return score


def parse_args():
    parser = argparse.ArgumentParser(description='ppd-aware clustering')
    parser.add_argument('--run_name', type=str, default=None, required=True,
                        help='wandb run name')
    parser.add_argument('--training_output_path', type=str, required=True,
                        help='location of training output')
    parser.add_argument('--n_hidden_fea', type=int, required=True,
                        help='number of hidden size in LSTM')
    
    parser.add_argument('--path_to_file_to_split', type=str, required=True,
                        help='location of input dataset')

    parser.add_argument('--path_to_labels', type=str, required=True,
                        help='location of labels')
    parser.add_argument('--test_size', type=float, default=0.33, help='percentage of total size for test split')
    parser.add_argument('--autoencoder_type', type=str, default=AutoEncoderEnum.ORIGINAL_DICE.value, required=False, choices=[i.value.lower() for i in AutoEncoderEnum],
                        help='auto encoder architecture')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--n_input_fea', type=int, required=True,
                        help='number of original input feature size')
    parser.add_argument('--n_dummy_demov_fea', type=int, required=True,
                        help='number of dummy demo feature size')
    parser.add_argument('--lstm_layer', type=int, default=1,
                        help='number of hidden size in LSTM')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lstm_dropout', type=float, default=0.0, help='dropout in LSTM')
    parser.add_argument('--K_clusters', type=int, required=True,
                        help='number of initial clusters')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--input_trained_data_train', type=str, required=False,
                        help='location of the data corpus')
    parser.add_argument('--input_trained_model', type=str, required=False,
                        help='location of the data corpus')
    parser.add_argument('--cuda', type=int, default=0,
                        help='If use cuda')
    args = parser.parse_args()
    return args

def calculate_rate_value(true_y, predict_results):
    conf_mat = confusion_matrix(true_y, predict_results)
    conf_mat_tolist = conf_mat.tolist()
    print("conf_mat_tolist=",conf_mat_tolist)
    number_TN = conf_mat_tolist[0][0]
    number_FN = conf_mat_tolist[1][0]
    number_FP = conf_mat_tolist[0][1]
    number_TP = conf_mat_tolist[1][1]
    acc=accuracy_score(true_y, predict_results)
    try:
        fpr=number_FP/(number_FP + number_TN)
        tpr=number_TP/(number_TP + number_FN)
        fnr=number_FN/(number_FN + number_TP)
        tnr=number_TN/(number_TN + number_FP)
        PPV=number_TP/(number_TP + number_FP) #positive and negative predictive values
        NPV=number_TN/(number_FN + number_TN)
        #message = 'acc: {:.4f}, fpr: {:.4f}, tpr: {:.4f}, fnr: {:.4f}, tnr: {:.4f}, PPV: {:.4f}, NPV: {:.4f}'.format(acc,fpr,tpr,fnr,tnr,PPV,NPV)
        #print(message)
        message = {'acc': acc, 'fpr':fpr, 'tpr':tpr, 'fnr':fnr, 'tnr':tnr, 'PPV':PPV, 'NPV':NPV}
    except:
        if number_FP + number_TN != 0:
            fpr=number_FP/(number_FP + number_TN)
        elif number_FP ==0:
            fpr = 0 
        else:
            fpr = 1 
        
        if (number_TP + number_FN) != 0:
            tpr=number_TP/(number_TP + number_FN)
        elif number_TP==0:
            tpr = 0
        else:
            trp = 1
        
        if (number_FN + number_TP) != 0:
            fnr=number_FN/(number_FN + number_TP)
        elif number_FN==0:
            fnr = 0
        else:
            fnr = 1 
        
        if (number_TN + number_FP)!=0:
            tnr=number_TN/(number_TN + number_FP)
        elif number_TN ==0:
            tnr = 0
        else:
            tnr = 1 
        
        if (number_TP + number_FP)!=0:
            PPV=number_TP/(number_TP + number_FP) #positive and negative predictive values
        elif number_TP ==0:
            PPV = 0
        else:
            PPV = 1 
        
        if (number_FN + number_TN)!=0:
            NPV=number_TN/(number_FN + number_TN)
        elif number_TN ==0:
            NPV = 0
        else:
            PPV = 1 
        message = {'acc': acc, 'fpr':fpr, 'tpr':tpr, 'fnr':fnr, 'tnr':tnr, 'PPV':PPV, 'NPV':NPV}
    return message 


def calculate_metrice(true_y, prediction_prob):
    pos = sum(true_y)
    neg = len(true_y) - pos
    true_y = np.array(true_y)
    prediction_prob = np.array(prediction_prob)
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(true_y, prediction_prob)
    #print("AUC=", metrics.auc(fpr, tpr))
    auc_score = metrics.auc(false_positive_rate, true_positive_rate)
    #print("auc_score=",auc_score)
    #print("-----------------")

    youden = []
    for i in range(len(thresholds)):
        thres = thresholds[i]
        fpr = false_positive_rate[i]
        tpr = true_positive_rate[i]
        youden.append(tpr - fpr)
    
    max_youden = max(youden)
    optim_thres = thresholds[youden.index(max_youden)]
    print("use optim_thres from auc=",optim_thres,", type(optim_thres)=",type(optim_thres))
    predict_results = [1 if p>=optim_thres else 0 for p in prediction_prob ]
    message1 = calculate_rate_value(true_y, predict_results) 
    message1["thres"]= optim_thres
    #print("-----------------")
    #print("use threshold 0.5:")
    predict_results = [1 if p>=0.5 else 0 for p in prediction_prob ]
    message2 = calculate_rate_value(true_y, predict_results)
    message2["thres"]= 0.5
    return auc_score, message1, message2  


def plot_roc(labels, predict_prob):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
    
    youden = []
    for i in range(len(thresholds)):
        thres = false_positive_rate[i]
        tpr = true_positive_rate[i]
        fpr = false_positive_rate[i]
        youden.append(tpr - fpr)
    
    max_youden = max(youden)
    optim_thres = thresholds[youden.index(max_youden)]
    print("optim_thres=",optim_thres)
    
    print("after use optim_thres")
    #print("predict_prob=",predict_prob)
    predict_results = [1 if p>=optim_thres else 0 for p in predict_prob ]
    #print("predict_results=",predict_results)
    conf_mat = confusion_matrix(labels, predict_results)
    print("conf_mat=",conf_mat)
    print("\nclassification_report=\n",classification_report(labels, predict_results))
    print("accuracy_score=",accuracy_score(labels, predict_results))

    return optim_thres

def compute_predictions_and_metrics(model, feature, target, split):

    print("------------------------")
    print(f"in {split} set")
    
    predict_results = model.predict(feature)
    predict_prob = model.predict_proba(feature)
    predict_prob1 = predict_prob[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(target, predict_prob1)
    print("auc=",metrics.auc(fpr, tpr))

    plot_roc(target, predict_prob1)
    auc_score, message1, message2   = calculate_metrice(target, predict_prob1)
    print("auc_score=", auc_score)
    print("message1=", message1)
    print("message2=", message2)

    return auc_score, message1, message2


if __name__ == '__main__':
    args = parse_args()

    #seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    wandb.init(project='DICE', name = args.run_name, config = args)

    print("(K,hn)=", args.K_clusters, args.n_hidden_fea)
    n_clusters, inputnhidden = args.K_clusters, args.n_hidden_fea
    taskpath = args.training_output_path
    args.input_trained_model = taskpath + 'hn_'+str(inputnhidden) +'_K_'+str(n_clusters)+'/part2_AE_nhidden_' + str(inputnhidden) + '/model_iter.pt'
    args.input_trained_data_train = taskpath + 'hn_'+str(inputnhidden) +'_K_'+str(n_clusters)+'/part2_AE_nhidden_' + str(inputnhidden) +'/data_train_iter.pickle'

    with open(args.path_to_file_to_split, 'rb') as handle:
        table = pickle.load(handle)
    y = pd.read_csv(args.path_to_labels)

    #table = table.sample(frac=0.10, random_state=args.seed)
    #y = y.sample(frac=0.10, random_state=args.seed)

    X_train, X_test, y_train, y_test = train_test_split(table, y, test_size=args.test_size, random_state=args.seed, shuffle=False)
    
    with open(args.input_trained_data_train, 'rb') as handle:
        data_train = pickle.load(handle)
    
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    data_test = yf_dataset_withdemo(X_test, y_test, args.n_hidden_fea, mode='test')
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    #data_valid = yf_dataset_withdemo(args.input_path, args.filename_valid, args.n_hidden_fea)
    #dataloader_valid = torch.utils.data.DataLoader(data_valid, batch_size=1, shuffle=False, drop_last=True)

    model = model_2(args.n_input_fea, args.n_hidden_fea, args.lstm_layer, args.lstm_dropout, args.K_clusters, args.n_dummy_demov_fea, args.cuda, args.autoencoder_type)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #print(model)
    if args.cuda:
        model = model.cuda()

    device = torch.device("cpu")
    model.load_state_dict(torch.load(args.input_trained_model, map_location=device))

    # update the rep and c first, then calculate the result. Here we evaluate the clustering metrics on data_test.
    update_curset_pred_C_and_repD0420(args, model, data_test, dataloader_test,"data_test", data_train)
    #update_curset_pred_C_and_repD0420(args, model, data_valid, dataloader_valid,"data_valid", data_train)
    
    feature_train = data_train.rep.numpy()
    target_train = np.array(data_train.data_y)
    feature_test = data_test.rep.numpy()#data_valid.rep.numpy()
    target_test = np.array(data_test.data_y)#np.array(data_valid.data_y)


    # power_ = 0
    # model =  LogisticRegression(C=10**power_, multi_class='multinomial', solver='lbfgs',max_iter=200)

    model = LogisticRegression(max_iter=200)
    model.fit(feature_train, target_train)

    auc_score, message1, message2 = compute_predictions_and_metrics(model, feature_train, target_train, 'train')
    
    train_recording = {}
    train_recording['train_auc'] = auc_score
    for res in message1:
        train_recording[f'train_optim_thres_{res}'] = message1[res]
    for res in message2:
        train_recording[f'train_default_thres_{res}'] = message2[res]

    auc_score, message1, message2 = compute_predictions_and_metrics(model, feature_test, target_test, 'test')
    test_recording = {}
    test_recording['test_auc'] = auc_score
    for res in message1:
        test_recording[f'test_optim_thres_{res}'] = message1[res]
    for res in message2:
        test_recording[f'test_default_thres_{res}'] = message2[res]

    train_recording.update(test_recording)
    rec_final = train_recording
    wandb.log(rec_final)