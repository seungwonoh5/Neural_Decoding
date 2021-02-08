# Real-Time Neural Decoder with Online Learning
This project proposes a novel learning method for training real-time neural decoders based on machine learning models that detect fine movements from neural activity signals extracted from animals' brains and continuously adapt to the data stream by exploiting knowledge from previous datasets in the stream. We show that after deploying the trainined model constantly updating the model to the future data streams could improve and maintain the model's performance. 

The dataset I have used for this experiment is private, so it cannot be shared. In order to use your own custom dataset, you should probably modify the file data.py for using your own.

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
To run the experiment, run ```python3 train.py.```

## What's Included
5 files with 4 scripts and 1 notebook file.
* data.py:this file provides all the data loading and preprocessing functions. (need to modify to use it for your own dataset)
* models.py: this file provides all of the decoder models in Keras. 
* utils.py: this file provides all the visualization and misc functions.
* train.py: this file serves as the main script for the experiment that trains both the decoder and the baseline model and compare the results.
* plot_results.ipynb: an example file that trains and plots the experimental results for visualization on Google Colab.

## Results
We perform extensive experiments on six datasets in a class-incremental setting, leading to significant improvements over the state of the art methods (e.g.,a 21.3% boost on CIFAR100 with 10 incremental tasks). Specifically, on large-scale datasets that generally prove difficult cases for incremental learning, our approach delivers absolute gains as high as 19.1% and 7.4% on ImageNetand MS-Celeb datasets, respectively.

## Contact
Author: Seungwon Oh - [aspiringtechsavvy@gmail.com](aspiringtechsavvy@gmail.com) or [soh1@terpmail.umd.edu](soh1@terpmail.umd.edu).a
To ask questions or report issues, please open an issue on the issues tracker. Discussions, suggestions and questions are welcome!
