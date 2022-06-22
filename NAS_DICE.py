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

from tqdm import tqdm
from DICE import yf_dataset_withdemo, model_2, func_analysis_test_error_D0406, collate_fn, update_M
import pickle
from sklearn.model_selection import train_test_split

def parse_args():
    # begin main function
    parser = argparse.ArgumentParser(description='ppd-aware clustering')
    # require para
    parser.add_argument('--run_name', type=str, default=None, required=True,
                        help='wandb run name')
    parser.add_argument('--init_AE_epoch', type=int, required=False,
                        help='number of epoch for representation initialization')
    parser.add_argument('--n_hidden_fea', type=int, required=False,
                        help='number of hidden size in LSTM')


    parser.add_argument('--path_to_file_to_split', type=str, required=True,
                        help='location of input dataset')

    parser.add_argument('--path_to_labels', type=str, required=True,
                        help='location of labels')
    parser.add_argument('--test_size', type=float, default=0.33, help='percentage of total size for test split')

    
    parser.add_argument('--training_output_path', type=str, required=True,
                        help='location of training output')
    
    parser.add_argument('--n_input_fea', type=int, required=True,
                        help='number of original input feature size')

    parser.add_argument('--n_dummy_demov_fea', type=int, required=True,
                        help='number of dummy demo feature size')
    parser.add_argument('--lstm_layer', type=int, default=1,
                        help='number of hidden size in LSTM')
    
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--lstm_dropout', type=float, default=0.0, help='dropout in LSTM')
    
    parser.add_argument('--K_clusters', type=int, required=False,
                        help='number of initial clusters')
    
    parser.add_argument('--iter', type=int, default=20,
                        help='maximum of iterations in iteration merge clusters')
    
    parser.add_argument('--epoch_in_iter', type=int, default=1,
                        help='maximum of iterations in iteration merge clusters')
    
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--input_trained_data_train', type=str, required=False,
                        help='location of the data corpus')
    
    parser.add_argument('--input_trained_model', type=str, required=False,
                        help='location of the data corpus')
    
    parser.add_argument('--cuda', type=int, default=0,
                        help='If use cuda')
    
    parser.add_argument('--lambda_AE', type=float, default=1.0, help='lambda of AE in iteration')
    parser.add_argument('--lambda_classifier', type=float, default=1.0, help='lambda_classifier of classifier in iteration')
    parser.add_argument('--lambda_outcome', type=float, default=10.0, help='lambda of outcome in iteration')
    parser.add_argument('--lambda_p_value', type=float, default=1.0, help='lambda of p value in iteration')
    args = parser.parse_args()
    return args

def func_analysis_test_error_D0420(args, model, data_test, dataloader_test):
    model.eval()
    criterion_MSE = nn.MSELoss()
    criterion_BCE = nn.BCELoss()
    error_AE = []
    error_outcome_likelihood = []
    correct = 0 
    total = 0 
    correct_outcome = 0
    outcome_auc = 0 
    outcome_true_y = []
    outcome_pred_prob = [] 
    for batch_idx, (index, batch_xvy, batch_c) in enumerate(dataloader_test):
        data_x, data_v, target = batch_xvy
        data_x = torch.autograd.Variable(data_x)
        data_v = torch.autograd.Variable(data_v)
        target = torch.autograd.Variable(target)
        batch_c = torch.autograd.Variable(batch_c)

        if args.cuda:
            data_x = data_x.cuda()
            data_v = data_v.cuda()
            target = target.cuda()
            batch_c = batch_c.cuda()
        
        encoded_x, decoded_x, output_c_no_activate, output_outcome = model(x=data_x, function="outcome_logistic_regression", demov=data_v)
        
        loss_AE = criterion_MSE(data_x, decoded_x)
        loss_outcome = criterion_BCE(output_outcome, target.float())
        error_outcome_likelihood.append(loss_outcome.data.cpu().numpy())
        error_AE.append(loss_AE.data.cpu().numpy())

        total += batch_c.size(0)

        outcome_true_y.append(target.data.cpu())
        outcome_pred_prob.append(output_outcome.data.cpu()) 
    
    test_AE_loss = np.mean(error_AE)
    test_outcome_likelihood = np.mean(error_outcome_likelihood)
    
    aucscore = outcome_auc_score = roc_auc_score(np.concatenate(outcome_true_y, 0), np.concatenate(outcome_pred_prob, 0))
    fpr, tpr, thresholds= roc_curve(np.concatenate(outcome_true_y, 0),  np.concatenate(outcome_pred_prob, 0))
    return test_AE_loss, test_outcome_likelihood,  outcome_auc_score, fpr, tpr, thresholds

def ppv_item(true_y, predict_results):
    acc=-1
    ppv=-1
    acc= accuracy_score(true_y, predict_results)
    conf_mat = confusion_matrix(true_y, predict_results)
    conf_mat_tolist = conf_mat.tolist()
    number_TN = conf_mat_tolist[0][0]
    number_FN = conf_mat_tolist[1][0]
    number_FP = conf_mat_tolist[0][1]
    number_TP = conf_mat_tolist[1][1]
    try:
        ppv=number_TP/(number_TP + number_FP) #positive and negative predictive values
    except:
        ppv=-1

    if number_FN == 0:
        FNR = 0
    else:
        FNR = number_FN/(number_TP+number_FN)
    if number_TP == 0:
        TPR = 0
    else:
        TPR = number_TP/(number_TP+number_FN)
    return acc, ppv ,TPR, FNR, conf_mat

def update_curset_pred_C_and_repD0420(args, model, data_cur, dataloader_cur, varname, datatrainM):
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
            final_embed[l.item()]=embed[l.item() % len(embed)]#final_embed[index] = embed

    data_cur.rep = final_embed 
    representations = data_cur.rep
    for i in range(data_cur.rep.size()[0]):
        embed = representations[i,:]
        trans_embed = embed.view(embed.size()+(1,))
        xj = torch.norm(trans_embed - datatrainM.M, dim=0)
        new_cluster = torch.argmin(xj)
        data_cur.C[i] = new_cluster


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


def change_label_from_highratio_to_lowratio(K, oldlabel, data_train):
    data_v = data_train.data_v
    data_y = data_train.data_y

    list_c = oldlabel.tolist()
    dict_c_count = {}
    dict_outcome_in_c_count = {}
    for i in range(K):
        dict_c_count[i] = 0 
        dict_outcome_in_c_count[i] = 0 
    
    for i in range(len(list_c)):
        dict_c_count[list_c[i]] += 1 
        if data_y[i]==1:
            dict_outcome_in_c_count[list_c[i]] += 1 

    dict_outcome_ratio = {}
    for keyc in dict_c_count:
        dict_outcome_ratio[keyc] = dict_outcome_in_c_count[keyc]/dict_c_count[keyc]
    
    sorted_dict_outcome_ratio = dict(sorted(dict_outcome_ratio.items(), key=lambda x:x[1], reverse=True))
    order = list(sorted_dict_outcome_ratio.keys())
    order_c_map = {}
    for i in range(len(order)):
        order_c_map[order[i]] = i
    # change c 
    new_list_c = []
    for i in range(len(list_c)):
        new_list_c.append(order_c_map[list_c[i]])
    
    return torch.LongTensor(new_list_c), order_c_map

def analysis_architecture(args, inputmodelpath, inputdatatrainpath, inputnhidden, n_clusters):
    args.input_trained_data_train = inputdatatrainpath
    args.input_trained_model = inputmodelpath
    args.n_hidden_fea = inputnhidden
    args.K_clusters = n_clusters

    # load data
    with open(args.path_to_file_to_split, 'rb') as handle:
        table = pickle.load(handle)
    y = pd.read_csv(args.path_to_labels)

    #table = table.sample(frac=0.10, random_state=args.seed)
    #y = y.sample(frac=0.10, random_state=args.seed)

    X_train, X_test, y_train, y_test = train_test_split(table, y, test_size=args.test_size, random_state=args.seed, shuffle=False)
    
    with open(args.input_trained_data_train, 'rb') as handle:
        data_train = pickle.load(handle)
    #data_train = yf_dataset_withdemo(X_train, y_train, args.n_hidden_fea, mode='train')
    #import IPython; IPython.embed()
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    data_test = yf_dataset_withdemo(X_test, y_test, args.n_hidden_fea, mode='test')
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    

    # Algorithm 2 model
    model = model_2(args.n_input_fea, args.n_hidden_fea, args.lstm_layer, args.lstm_dropout, args.K_clusters, args.n_dummy_demov_fea, args.cuda)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model = model.cuda()

    device = torch.device("cpu")
    model.load_state_dict(torch.load(args.input_trained_model, map_location=device))
    # model.to(device)
    train_AE_loss, _, train_outcome_likelihood, train_outcome_auc_score = func_analysis_test_error_D0406(args, model, data_train, dataloader_train)
    #valid_AE_loss, valid_outcome_likelihood, valid_outcome_auc_score, fpr, tpr, thresholds = func_analysis_test_error_D0420(args, model, data_valid, dataloader_valid)
    test_AE_loss, _, test_outcome_likelihood,  test_outcome_auc_score = func_analysis_test_error_D0406(args, model, data_test, dataloader_test)

    #update_curset_pred_C_and_repD0420(args, model, data_valid, dataloader_valid,"data_valid", data_train)
    update_curset_pred_C_and_repD0420(args, model, data_test, dataloader_test,"data_test", data_train)
    dict_outcome_ratio_train, dict_c_count = analysis_cluster_number_byclustering(data_train, n_clusters, 0, "train")
    #dict_outcome_ratio_valid,_ = analysis_cluster_number_byclustering(data_valid, n_clusters, 0, "valid")
    dict_outcome_ratio_test,_ = analysis_cluster_number_byclustering(data_test, n_clusters, 0, "test")

    
    #resx = np.concatenate([data_train.rep.numpy(), data_test.rep.numpy()], axis=0) #resx = np.concatenate([data_train.rep.numpy(), data_valid.rep.numpy(), data_test.rep.numpy()], axis=0)
    #resy = data_train.data_y + data_test.data_y #resy = data_train.data_y + data_valid.data_y + data_test.data_y
    #resc = torch.cat((data_train.C, data_test.C),0) #resc = torch.cat((data_train.C, data_valid.C, data_test.C),0)
    
    trainx, trainy, trainc = data_train.rep.numpy(), data_train.data_y, data_train.C
    return data_train, data_test, trainx, trainy, trainc, dict_outcome_ratio_train, dict_c_count,test_outcome_likelihood, test_outcome_auc_score
    #return data_train, data_valid, data_test, resx , resy , resc , trainx, trainy, trainc, dict_outcome_ratio_train, dict_c_count,test_outcome_likelihood, test_outcome_auc_score, valid_outcome_likelihood, valid_outcome_auc_score


import os
import re 
if __name__ == '__main__':
    
    args = parse_args()

    wandb.init(project='DICE', name = args.run_name, config = args)

    #seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    path = "./" 
    files= os.listdir(path) 
    s = []
    record = [] 

    print("useful architecture: ")
    for file in sorted(files):
        if file[-4:]!='.log':
            continue
        _,K,hn,_ = re.split('k|hn|.log', file)
        list_line = []
        f = open(path+"/"+file)
        iter_f = iter(f); 
        for line in iter_f: 
            list_line.append(line)
        myline_index= -2
        if '[]' not in list_line[myline_index]:
            lastiter=re.split('\[|,|\]',list_line[myline_index])[-2]
            print("# file={}, K={}, hn={}, last iter={}".format(file, K, hn, lastiter))
            record.append((int(K),int(hn),int(lastiter)))
    print('record=',(record))
    print('len(record)=', len(record))

    list_tuple_k_hn_iter_nll_auc_biggestratio_valid = []
    list_tuple_k_hn_iter_nll_auc_biggestratio_test = []
    list_tuple_k_hn_iter_dict_outcome_ratio_train = []
    for item in record:
        n_clusters, inputnhidden, epoch = item
        taskpath = './'
        inputmodelpath = taskpath + 'hn_'+str(inputnhidden) +'_K_'+str(n_clusters)+'/part2_AE_nhidden_' + str(inputnhidden) + '/model_iter.pt'
        inputdatatrainpath = taskpath + 'hn_'+str(inputnhidden) +'_K_'+str(n_clusters)+'/part2_AE_nhidden_' + str(inputnhidden) +'/data_train_iter.pickle'
        data_train, data_test, trainx, trainy, trainc, dict_outcome_ratio_train, dict_c_count,test_outcome_likelihood, test_outcome_auc_score = analysis_architecture(args, inputmodelpath, inputdatatrainpath, inputnhidden,n_clusters) #, valid_outcome_likelihood, valid_outcome_auc_score
        if 0 in list(dict_c_count.values()):
            print("degenerated clusters. Some clusters has no pid")
            continue
        from sklearn import manifold, datasets
        #list_tuple_k_hn_iter_nll_auc_biggestratio_valid.append((n_clusters, inputnhidden, epoch, valid_outcome_likelihood, valid_outcome_auc_score, dict_outcome_ratio_train[0]))
        list_tuple_k_hn_iter_nll_auc_biggestratio_test.append((n_clusters, inputnhidden, epoch, test_outcome_likelihood, test_outcome_auc_score, dict_outcome_ratio_train[0]))
        list_tuple_k_hn_iter_dict_outcome_ratio_train.append((n_clusters, inputnhidden, epoch, dict_outcome_ratio_train))

    print("--------- valid ---------")
    min_nll= sorted( list_tuple_k_hn_iter_nll_auc_biggestratio_test, key=lambda x:x[3], reverse=False)
    print("min_nll=",min_nll)
    print()

    max_AUC= sorted( list_tuple_k_hn_iter_nll_auc_biggestratio_test, key=lambda x:x[4], reverse=True)
    print("max_AUC=",max_AUC)
    print()

    max_biggestratio= sorted( list_tuple_k_hn_iter_nll_auc_biggestratio_test, key=lambda x:x[5], reverse=True)
    print("max_biggestratio=",max_biggestratio)
    print()
    print("final search result based on the maximum AUC score on validation set, K={}, hn={}".format(max_AUC[0][0], max_AUC[0][1]))

    
    wandb.log({
        'K': max_AUC[0][0], 
        'hn': max_AUC[0][1]
    })
