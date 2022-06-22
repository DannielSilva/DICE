#!/bin/bash

python representation_visualization.py --image_name 'tsne_3d_original.png' --cuda 1 --batch_size=4 --training_output_path "./" --path_to_file_to_split='../diabetes_processedFeats_orderedSeqLength.pickle' --path_to_labels='../y_diabetes.csv'  --n_input_fea 3035 --n_dummy_demov_fea 2 --K_clusters 2 --n_hidden_fea 80