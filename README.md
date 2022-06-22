## Our Paper Title

This repository is the implementation of "DICE: Deep Significance Clustering for
Outcome-Driven Stratification". 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Input data
Input datatrain.pkl, datavalid.pkl, datatest.pkl for training set, validation set and test set, respectively. 

Data format:
datatrain: List[ data_x, data_v, data_y] </br>
data_x: List[List[list[float]]] of shape [data_size, seq_len, n_input_fea]</br>
data_v: List[List[float]] of shape [data_size, n_dummy_demov_fea]</br>
data_y: List[int] of shape [data_size] </br>

where n_input_fea and n_dummy_demov_fea are the size of features at each timestamp and demographic features, respectively. </br>



## Training

To train the model(s) in the paper, run the following commands:

```train
./run_DICE.sh
```

Here in each bash file, we train the model with fixed K (number of clusters) and d ( dimension of representation), that is: 
```
python DICE.py --run_name="n_hidden_fea $n_hidden_fea for k2" --cuda 1 --batch_size=4 --init_AE_epoch 1 --n_hidden_fea  $n_hidden_fea --path_to_file_to_split='../all_data.pickle' --path_to_labels='../y.csv' --n_input_fea 3035 --n_dummy_demov_fea 2 --lstm_layer 1 --lr 0.0001 --K_clusters 2 --iter 60 --epoch_in_iter 1 --lambda_AE 1.0 --lambda_classifier 1.0 --lambda_outcome 10.0 --lambda_p_value 1.0 >k2hn$n_hidden_fea.log
```
Parameters:
```
      --run_name : wandb run name
      --path_to_file_to_split : data file with data to split in train and test splits
      --path_to_labels : path to label csv
      --init_AE_epoch: epoch for the representation initialization by AutoEncoder
      --n_hidden_fea: the dimension for representation, hn for short in the below content
      --n_input_fea: dimension of input file features 
      --n_dummy_demov_fea: number of features of v (dummy features of demographics) 
      --lr: learning rate 
      --K_clusters: number of clusters 
      --iter: interation of algorithm 
      --epoch_in_iter: number of epoch in each iteration 
      --lambda_AE: weight of AutoEncoder reconstration loss 
      --lambda_classifier: weight of cluster membership classifier loss 
      --lambda_outcome: weight of outcome classifier loss 
      --lambda_p_value: weight of significance difference constraint loss
```
Output: 
``` 
      For each architer network (K, hn),  there would be a log file named k*hn*.log and a folder named 
      hn_*_K_*. The output model and datatrain with updated representation are stored in 
      ./hn_*_K_*/part2_AE_hnidden_*/model_iter.pt and data_train_iter.pickle, respectively.
```


## Architecture search
After training, we do the neural architecture search based on the AUC score on validation set,
``` 
./run_NAS_DICE.sh
```
which contains:

``` architecture_search
python NAS_DICE.py --cuda 1 --batch_size=4 --training_output_path "./" --path_to_file_to_split='../all_data.pickle' --path_to_labels='../y.csv'  --n_input_fea 3035 --n_dummy_demov_fea 2
```
Parameters:
```
      --run_name : wandb run name
      --path_to_file_to_split : data file with data to split in train and test splits
      --path_to_labels : path to label csv
      --training_output_path: path of training output 
      --n_input_fea: dimension of input file features 
      --n_dummy_demov_fea: number of features of v (dummy features of demographics) 
```
Output: 
```
      We can find the best K and hn according to the last line of the log, such as:
final search result based on the maximum AUC score on validation set, K=4, hn=35
```

## Evaluation
### 1. Visualization of representations, run 

``` 
./run_representation_visualization.sh
```
which contains:

```visualization
python representation_visualization.py --image_name 'tsne_3d_original.png' --cuda 1 --batch_size=4 --training_output_path "./" --path_to_file_to_split='../all_data.pickle' --path_to_labels='../y.csv'  --n_input_fea 3035 --n_dummy_demov_fea 2 --K_clusters 2 --n_hidden_fea 80
```
Parameters: 
```
      --training_output_path: path of training output
      --image_name : output image name
      --path_to_file_to_split : data file with data to split in train and test splits
      --path_to_labels : path to label csv
      --n_input_fea: dimension of input file features 
      --n_dummy_demov_fea: number of features of v (dummy features of demographics) 
      --K_clusters: number of clusters
      --n_hidden_fea: the dimension for representation
```
Output: Figure named after args.image_name.</br>


### 2. Clustering performance on test set, run 
``` 
./run_clustering_metrics.sh
```
which contains:
```clustering
python clustering_metrics.py --run_name='clustering-metrics-1' --cuda 1 --batch_size=4 --training_output_path "./" --path_to_file_to_split='../all_data.pickle' --path_to_labels='../y.csv'  --n_input_fea 3035 --n_dummy_demov_fea 2 --K_clusters 2 --n_hidden_fea 80
```
Parameters: the same as the former subsection. </br>
Output: Silhouette score, Calinski-Harabasz score, Davies_Bouldin score. </br>


### 3. Outcome prediction with representation as input, run
``` 
./run_outcome_prediction.sh
```
which contains:
``` outcome_prediction 
python outcome_prediction.py --run_name='outcome-prediction' --cuda 1 --batch_size=4 --training_output_path "./" --path_to_file_to_split='../diabetes_processedFeats_orderedSeqLength.pickle' --path_to_labels='../y_diabetes.csv'  --n_input_fea 3035 --n_dummy_demov_fea 2 --K_clusters 2 --n_hidden_fea 80
```
Parameters: the same as the former subsection. </br>
Output: AUC, ACC, FPR, TPR, FNR, TNR, PPV, NPV. </br>