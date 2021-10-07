# Adaptive Real-Time Neural Decoder with Online Learning
This project experiments different learning methods for LSTM-based real-time neural decoder that detect fine movements from neural activity signals extracted from animals' brains and continuously adapt to the data stream in an online setting. We show that a combination of transfer learning that develop source model from other datasets and online learning that constantly updates the model to future data streams improve and better maintain the trained model's performance after deployment compared to batch learning.

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

## Installation
Clone this repository by running ```git clone https://github.com/seungwonoh5/Online_Learning_Neural_Decoding```.

## Dependencies
To run our program, it requires the following:
* numpy==1.19.5
* pandas==1.1.5
* matplotlib==3.2.2
* tensorflow==2.4.1
* scikit-learn==0.22.2

All packages will be installed automatically when running ```pip3 install -r requirements.txt``` to install all the dependencies.

## What's Included
Inside the repo, there are 4 scripts and 1 notebook file.
* data.py: this file provides all the data loading and preprocessing functions. (need to modify to use it for your own dataset)
* models.py: this file provides all the decoder models in Keras. 
* utils.py: this file provides all the visualization and misc functions.
* main.py: this file serves as the main script for the experiment that trains both the decoder and the baseline model and compare the results.
* main.ipynb: an example file that trains and plots the experimental results for visualization on Google Colab.

## Getting Started
We have included jupyter notebooks that provide detailed examples of our experiments.
You can either run the notebook ```main.ipynb``` which goes through the complete process of the experiment and outputs visualizations of the experiment or
run main script that imports other script modules in the repo to goes through the whole experiment. You can use individual scripts to reuse part of the program that you need.
```
python main.py
```

## Results
We perform extensive experiments on six datasets sequentially streaming and we show that an online setting continuously updating the model as every data block is processed leads to significant improvements over various state of the art models compared to the batch learning method that the model is fixed after training on the initial dataset and deploying for prediction. Specifically, on large-scale datasets that generally prove difficult cases for incremental learning, our approach delivers absolute gains as high as 19.1% and 7.4% on datasets, respectively.

## Contact
Author: Seungwon Oh - [aspiringtechsavvy@gmail.com](aspiringtechsavvy@gmail.com) or [soh1@terpmail.umd.edu](soh1@terpmail.umd.edu).
To ask questions or report issues, please open an issue on the issues tracker. Discussions, suggestions and questions are welcome!
