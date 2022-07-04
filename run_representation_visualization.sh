#!/bin/bash

training_output_path="original/" #dont forget the bar
K_clusters=3
n_hidden_fea=60
name='original'

#image_name building
image_name='tsne_3d_'
image_name+=$name
image_name+='_'
image_name+=$K_clusters
image_name+='_'
image_name+=$n_hidden_fea
image_name+='.png'


echo $image_name
python representation_visualization.py --image_name=$image_name --training_output_path=$training_output_path --cuda 1 --batch_size=4 --path_to_file_to_split='../diabetes_processedFeats_orderedSeqLength.pickle' --path_to_labels='../y_diabetes.csv'  --n_input_fea 3035 --n_dummy_demov_fea 2 --K_clusters $K_clusters --n_hidden_fea $n_hidden_fea