#!/bin/bash
K_clusters=3
n_hidden_fea=60
run_name='original-clustering-metrics'
training_output_path='original/' #dont forget the bar
autoencoder_type='original'

python clustering_metrics.py --run_name=$run_name --training_output_path=$training_output_path --cuda 1 --batch_size=4 --path_to_file_to_split='../diabetes_processedFeats_orderedSeqLength.pickle' --path_to_labels='../y_diabetes.csv'  --n_input_fea 3035 --n_dummy_demov_fea 2 --K_clusters $K_clusters --n_hidden_fea $n_hidden_fea --autoencoder_type=$autoencoder_type