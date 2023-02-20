#!/usr/bin/env python
import numpy as np 
import pickle
import torch

import argparse
import random

import pandas as pd
import numpy as np 

from sklearn import metrics
from DICE import yf_dataset_withdemo, model_2, collate_fn
from autoencoder_builder import AutoEncoderEnum

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
    for batch_idx, (index, data_x, data_v, target, batch_c) in enumerate(dataloader_cur):

        if args.cuda:
            data_x = data_x.cuda()
            data_v = data_v.cuda()
            target = target.cuda()

        encoded_x, decoded_x, output_c_no_activate, output_outcome = model(x=data_x, function="outcome_logistic_regression", demov=data_v)
        embed = encoded_x.data.cpu()
        for l in index:
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
    parser.add_argument('--autoencoder_type', type=str, default=AutoEncoderEnum.ORIGINAL_DICE.value, required=False, choices=[i.value.lower() for i in AutoEncoderEnum],
                        help='auto encoder architecture')
    parser.add_argument('--path_to_data_test', type=str, required=True,
                        help='location of input dataset')
    parser.add_argument('--path_to_labels_test', type=str, required=True,
                        help='location of input dataset')
    parser.add_argument('--test_size', type=float, default=0.33, help='percentage of total size for test split')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
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

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="5"

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
    
    X_test = pd.read_pickle(args.path_to_data_test)
    y_test = pd.read_pickle(args.path_to_labels_test)
    
    with open(args.input_trained_data_train, 'rb') as handle:
        data_train = pickle.load(handle)

    args.n_input_fea = data_train.data_x[0].shape[1]
    print('hidden features:', args.n_input_fea )
    
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    data_test = yf_dataset_withdemo(X_test, y_test, args.n_hidden_fea, mode='test')
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = model_2(args.n_input_fea, args.n_hidden_fea, args.lstm_layer, args.lstm_dropout, args.K_clusters, args.n_dummy_demov_fea, args.cuda, args.autoencoder_type)

    #print(model)
    if args.cuda:
        model = model.cuda()

    device = torch.device("cpu")
    model.load_state_dict(torch.load(args.input_trained_model, map_location=device))

    # update the rep and c first, then calculate the result. Here we evaluate the clustering metrics on data_test.
    update_curset_pred_C_and_repD0420(args, model, data_test, dataloader_test,"data_test", data_train)
    score_dice = calculate_cluster_metrics(data_test)

    for key in score_dice:
        print("{:<30}, dice score ={:.4f}".format(key, score_dice[key]))

    wandb.log(score_dice)

            