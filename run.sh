#!/bin/bash

# just one run by chosing hyperparameters of hidden vector size and number of clusters

set -e
maxnn=102
output_path='original-icd9/' #dont forget the bar "/" and create this directory first
data_path='../data-icd9/'
autoencoder_type='original'

n_hidden_fea=90
K_clusters=4

echo "process n_hidden_fea $n_hidden_fea for k$K_clusters "
python DICE.py --run_name="$output_path n_hidden_fea $n_hidden_fea for k$K_clusters" --cuda 1 --batch_size=24 --init_AE_epoch 1 --n_hidden_fea  $n_hidden_fea --path_to_data_train=$data_path'data_train_11_fp32.pickle' --path_to_data_test=$data_path'data_test_11_fp32.pickle' --path_to_labels_train=$data_path'y_train_11_fp32.pickle' --path_to_labels_test=$data_path'y_test_11_fp32.pickle' --n_dummy_demov_fea 2 --lstm_layer 1 --lr 0.0001 --K_clusters $K_clusters --iter 60 --epoch_in_iter 1 --lambda_AE 1.0 --lambda_classifier 1.0 --lambda_outcome 10.0 --lambda_p_value 1.0 --output_path=$output_path --autoencoder_type=$autoencoder_type > $output_path/k2hn$n_hidden_fea.log
