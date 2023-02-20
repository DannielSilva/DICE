#!/bin/bash

run_name='original-icd9-nas-dice-test'
training_output_path="original-icd9/" #dont forget the bar
autoencoder_type='original'
data_path='../data-icd9/'

python NAS_DICE.py --run_name=$run_name --cuda 1 --batch_size=16 --training_output_path=$training_output_path --path_to_data_test=$data_path'data_test_11_fp32.pickle' --path_to_labels_test=$data_path'y_test_11_fp32.pickle' --n_dummy_demov_fea 2 --autoencoder_type=$autoencoder_type