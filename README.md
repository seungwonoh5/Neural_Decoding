# Real-Time Neural Decoder: based on Online Transfer Learning
This project is to propose a novel learning method for training real-time neural decoder models that continuously adapts to the data stream of neural activities extracted from mice to exploit knowledge from previous datasets in the stream. We show that after deploying the trainined model constantly updating the model to the future data streams could improve and maintain the model's performance. 

## What's New
v1.5 (Oct 12)
* add features for the dummy neuron: average of all neuron values in that frame
* handle outliers in the data and measure similarity for dataset distribution
* try ways to improve f-1 score for continual transfer learning
* split into train/val and test and do cross-validation on train/val

v1.4 (~Sep 30)
* find why batch_transfer and continuous_transfer have dissimilar results
* hyperparameter optimization, iterating through a large number of settings (function *nn_grid_search*)
* use pca to preprocess (function *load_data*)

v1.3 (~Sep 16)
* implemented transfer learning framework (function *transfer_train*) 
* fine-tune (freeze none or some layers) vs. feature extractor (freeze all layers but last)

v1.2 (~Sep 2)
* implemented batch learning framework in *train* (function *batch_train*)
* transfer vs. non-transfer
* grid searched for SVM optimization in *build_svm*

v1.1 (~Aug 19)
* modularized code by writing in main script-function hierarchy
* designed 3-step pipeline(preprocess - train - visualize) to quickly test and compare many ML methods
* preprocessed 2D data into 3D for LSTM decoders(function *prep_recurrent_data*)
* build models for SVM, MLP, LSTM (function *build_{svm, mlp, lstm}*)
* visualized learning curve of decoders (function *visualize_train*)
* visualized prediction vs. ground truth (function *visualize_pred*)

v1.0 (~Aug 5)
* load and stack multiple datasets in 2D (function *load_data*)
* train and evaluate models using Keras (function *train*)

## Dependencies
This code requires the following:
* numpy==1.19.5
* pandas==1.1.5
* matplotlib==3.2.2
* tensorflow==2.4.1
* scikit-learn==0.22.2

run ```pip3 install -r requirements.txt``` to install all the dependencies.

## Usage
To run the experiment, run ```CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py [x].```
Here [x] is a system argument of the starting task id. begins with 0.

## Results


## Contact
Author: Seungwon Oh - [aspiringtechsavvy@gmail.com](aspiringtechsavvy@gmail.com) or [soh1@terpmail.umd.edu](soh1@terpmail.umd.edu).a
To ask questions or report issues, please open an issue on the issues tracker. Discussions, suggestions and questions are welcome!
