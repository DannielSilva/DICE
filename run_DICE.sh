#!/bin/bash
set -e
maxnn=102
output_path='original-icd9/' #dont forget the bar "/" and create this directory first
data_path='../data-icd9/'
autoencoder_type='original'

# K=2
for (( n_hidden_fea = 30; n_hidden_fea< $maxnn; n_hidden_fea+=10)) 
do
    echo "process n_hidden_fea $n_hidden_fea for k2 "
	python DICE.py --run_name="$output_path n_hidden_fea $n_hidden_fea for k2" --cuda 1 --batch_size=30 --init_AE_epoch 1 --n_hidden_fea  $n_hidden_fea --path_to_data_train=$data_path'data_train_11_fp32.pickle' --path_to_data_test=$data_path'data_test_11_fp32.pickle' --path_to_labels_train=$data_path'y_train_11_fp32.pickle' --path_to_labels_test=$data_path'y_test_11_fp32.pickle' --n_dummy_demov_fea 2 --lstm_layer 1 --lr 0.0001 --K_clusters 2 --iter 60 --epoch_in_iter 1 --lambda_AE 1.0 --lambda_classifier 1.0 --lambda_outcome 10.0 --lambda_p_value 1.0 --output_path=$output_path --autoencoder_type=$autoencoder_type > $output_path/k2hn$n_hidden_fea.log
done

# K=3
for (( n_hidden_fea = 30; n_hidden_fea< $maxnn; n_hidden_fea+=10)) 
do
    echo "process n_hidden_fea $n_hidden_fea for k3 "
    python DICE.py --run_name="$output_path n_hidden_fea $n_hidden_fea for k3"  --cuda 1 --batch_size=30 --init_AE_epoch 1 --n_hidden_fea  $n_hidden_fea --path_to_data_train=$data_path'data_train_11_fp32.pickle' --path_to_data_test=$data_path'data_test_11_fp32.pickle' --path_to_labels_train=$data_path'y_train_11_fp32.pickle' --path_to_labels_test=$data_path'y_test_11_fp32.pickle' --n_dummy_demov_fea 2 --lstm_layer 1 --lr 0.0001 --K_clusters 3 --iter 60 --epoch_in_iter 1 --lambda_AE 1.0 --lambda_classifier 1.0 --lambda_outcome 10.0 --lambda_p_value 1.0 --output_path=$output_path --autoencoder_type=$autoencoder_type > $output_path/k3hn$n_hidden_fea.log
done 

# K=4
for (( n_hidden_fea = 30; n_hidden_fea< $maxnn; n_hidden_fea+=10)) 
do
    echo "process n_hidden_fea $n_hidden_fea for k4 "
    python DICE.py  --run_name="$output_path n_hidden_fea $n_hidden_fea for k4" --cuda 1 --batch_size=30 --init_AE_epoch 1 --n_hidden_fea  $n_hidden_fea --path_to_data_train=$data_path'data_train_11_fp32.pickle' --path_to_data_test=$data_path'data_test_11_fp32.pickle' --path_to_labels_train=$data_path'y_train_11_fp32.pickle' --path_to_labels_test=$data_path'y_test_11_fp32.pickle' --n_dummy_demov_fea 2 --lstm_layer 1 --lr 0.0001 --K_clusters 4 --iter 60 --epoch_in_iter 1 --lambda_AE 1.0 --lambda_classifier 1.0 --lambda_outcome 10.0 --lambda_p_value 1.0 --output_path=$output_path --autoencoder_type=$autoencoder_type > $output_path/k4hn$n_hidden_fea.log
done 

K=5
for (( n_hidden_fea = 30; n_hidden_fea< $maxnn; n_hidden_fea+=10)) 
do
    echo "process n_hidden_fea $n_hidden_fea for k5 "
    python DICE.py  --run_name="$output_path n_hidden_fea $n_hidden_fea for k5" --cuda 1 --batch_size=30 --init_AE_epoch 1 --n_hidden_fea  $n_hidden_fea --path_to_data_train=$data_path'data_train_11_fp32.pickle' --path_to_data_test=$data_path'data_test_11_fp32.pickle' --path_to_labels_train=$data_path'y_train_11_fp32.pickle' --path_to_labels_test=$data_path'y_test_11_fp32.pickle' --n_dummy_demov_fea 2 --lstm_layer 1 --lr 0.0001 --K_clusters 5 --iter 60 --epoch_in_iter 1 --lambda_AE 1.0 --lambda_classifier 1.0 --lambda_outcome 10.0 --lambda_p_value 1.0 --output_path=$output_path --autoencoder_type=$autoencoder_type > $output_path/k5hn$n_hidden_fea.log
done 

