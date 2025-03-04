#!/usr/bin/env python
import numpy as np 
import pickle

import torch

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt 
import argparse

import random

import pandas as pd
import statsmodels.api as sm
import numpy as np 
import statsmodels.api as sm
from sklearn.metrics import auc, roc_auc_score, roc_curve


import matplotlib.pyplot as plt
from sklearn import manifold, datasets

from DICE import yf_dataset_withdemo

# def analysis_cluster_number_byclustering(data_cur, num_clusters, if_check, varname):
#     data_C = data_cur.C
#     data_v = data_cur.data_v
#     data_y = data_cur.data_y

#     list_c = data_C.tolist()
#     list_onehot = []
#     dict_c_count = {}
#     dict_outcome_in_c_count = {}
#     for i in range(num_clusters):
#         dict_c_count[i] = 0 
#         dict_outcome_in_c_count[i] = 0 
    
#     for i in range(len(list_c)):
#         temp = [0 for i in range(num_clusters)]
#         temp[list_c[i]] = 1 
#         list_onehot.append(temp)

#         dict_c_count[list_c[i]] += 1 
#         if data_y[i]==1:
#             dict_outcome_in_c_count[list_c[i]] += 1 
    
#     if if_check:
#         print("--------")
#         print("num_clusters=", num_clusters)
#         print()
#         print("list_c[0]=",list_c[0])
#         print("list_onehot[0]=", list_onehot[0])
#         print()
#         print("list_c[1]=",list_c[1])
#         print("list_onehot[1]=", list_onehot[1])
#         print("--------")
    
#     dict_outcome_ratio = {}
#     for keyc in dict_c_count:
#         if dict_c_count[keyc] == 0:
#             dict_outcome_ratio[keyc] = 0
#         else:
#             dict_outcome_ratio[keyc] = dict_outcome_in_c_count[keyc]/dict_c_count[keyc]
    
#     return dict_outcome_ratio, dict_c_count

def analysis_cluster_number_byclustering(data_cur, num_clusters, if_check, varname):
    print('new func')
    d_c = {k:0 for k in range(num_clusters)}
    
    c_arr = np.array(data_cur.C)
    y_arr = data_cur.data_y

    for c,y in zip(c_arr, y_arr):
        if y ==1:
            d_c[c] +=1
    dict_outcome = d_c

    unique, counts = np.unique(c_arr, return_counts=True)
    dict_c_count = {c:cnt for c,cnt in zip(unique,counts)}
    dict_outcome_ratio = {c:dict_outcome[c] / dict_c_count[c] for c in dict_c_count}
    return dict_outcome_ratio, dict_c_count

def parse_args():
    parser = argparse.ArgumentParser(description='ppd-aware clustering')
    parser.add_argument('--training_output_path', type=str, required=True,
                        help='location of training output')
    parser.add_argument('--n_hidden_fea', type=int, required=True,
                        help='number of hidden size in LSTM')
    parser.add_argument('--image_name', type=str, default=None, required=True,
                        help='result image file name')
    parser.add_argument('--test_size', type=float, default=0.33, help='percentage of total size for test split')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--n_dummy_demov_fea', type=int,
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
    parser.add_argument('--tsne_components', type=int, default=3,
                        help='number of components for tsne')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    #seeds
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)

    print("(K,hn)=", args.K_clusters, args.n_hidden_fea)
    n_clusters, inputnhidden = args.K_clusters, args.n_hidden_fea
    taskpath = args.training_output_path
    args.input_trained_model = taskpath + 'hn_'+str(inputnhidden) +'_K_'+str(n_clusters)+'/part2_AE_nhidden_' + str(inputnhidden) + '/model_iter.pt'
    args.input_trained_data_train = taskpath + 'hn_'+str(inputnhidden) +'_K_'+str(n_clusters)+'/part2_AE_nhidden_' + str(inputnhidden) +'/data_train_iter.pickle'

    with open(args.input_trained_data_train, 'rb') as handle:
        data_train = pickle.load(handle)
    dict_outcome_ratio_train, dict_c_count = analysis_cluster_number_byclustering(data_train, n_clusters, 0, "train")
    print(dict_outcome_ratio_train, dict_c_count)
    X, y, c = data_train.rep.numpy(), data_train.data_y, data_train.C
    import IPython; IPython.embed()
    
    print('tsne cmomp=', args.tsne_components)
    tsne = manifold.TSNE(n_components=args.tsne_components, random_state=888)
    X_tsne = tsne.fit_transform(X)
    print("X.shape=", X.shape)
    print("X_tsne.shape=", X_tsne.shape)
    total = X.shape[0]

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  

    # figure2
    numK = n_clusters
    c_k_dict = {}
    for i in range(numK):
        c_k_dict[i] = [] 

    for i in range(len(y)):
        curk = c[i].item()
        c_k_dict[curk].append(X_tsne[i,:])

    for key in c_k_dict:
        c_k_dict[key] = np.concatenate([x.reshape((1,args.tsne_components)) for x in c_k_dict[key]], axis=0)
        print(c_k_dict[key].shape)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    colorlist=['red','orange','blue','green','cyan','purple']
    for key in c_k_dict:
        if args.tsne_components == 3:
            ax.scatter(c_k_dict[key][:,0],c_k_dict[key][:,1],c_k_dict[key][:,2],s=30,color=colorlist[key],marker='.',alpha=0.5,label='cluster '+str(key+1)+ ', '+ str(round(dict_outcome_ratio_train[key]*100,2))+'% of outcome 1; ' + str(round( (dict_c_count[key] / total) * 100, 2)) + ' % of total patients') 
        elif args.tsne_components == 2:
            ax.scatter(c_k_dict[key][:,0],c_k_dict[key][:,1],s=30,color=colorlist[key],marker='.',alpha=0.5,label='cluster '+str(key+1)+ ', '+ str(round(dict_outcome_ratio_train[key]*100,2))+'% of outcome 1; ' + str(round( (dict_c_count[key] / total) * 100, 2)) + ' % of total patients') 

    plt.legend(fontsize = 14, bbox_to_anchor=(0.8, 0.1), loc="lower right")
    ax.view_init(elev=-73, azim= -0)
    ax.set_xlim(-16, 12)
    plt.savefig(args.image_name, bbox_inches='tight')
    plt.show()