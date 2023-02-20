#!/bin/bash

training_output_path="original-icd9/" #dont forget the bar
K_clusters=4
n_hidden_fea=90
autoencoder_type='original'
run_name='original-icd9-coutcome-prediction'
data_path='../data-icd9/'

python outcome_prediction.py --run_name=$run_name --training_output_path=$training_output_path --cuda 1 --batch_size=4 --path_to_data_test=$data_path'data_test_11_fp32.pickle' --path_to_labels_test=$data_path'y_test_11_fp32.pickle' --n_dummy_demov_fea 2 --K_clusters $K_clusters --n_hidden_fea $n_hidden_fea --autoencoder_type=$autoencoder_type