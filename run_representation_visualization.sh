#!/bin/bash

training_output_path="original-icd9/" #dont forget the bar
K_clusters=4 #3
n_hidden_fea=90 #60
name='original-icd9'
tsne_components=3

#image_name building
image_name='tsne_'$tsne_components'd_'$name'_'$K_clusters'_'$n_hidden_fea'.png'

echo $image_name
python representation_visualization.py --image_name=$image_name --training_output_path=$training_output_path --cuda 1 --K_clusters $K_clusters --n_hidden_fea $n_hidden_fea --tsne_components=2