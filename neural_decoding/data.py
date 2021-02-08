# -*- coding: utf-8 -*-
"""data.ipynb"""

import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

class NeuralData:
    def __init__(self, root_dir, subject, day, inj):
        """
        # for each dataset (=3000 data points)
        # load each dataset recorded on different days and injection conditions

        path: path to the csv file with annotations
        data_pts: list[tuple], list of tuples that each tuple represents a dataset with a specific day and injection condition
        window_size: int, the number of bins you want to use together to predict y
        """
        self.root_dir = root_dir
        self.subject = subject
        self.day = day
        self.inj = inj


    def load(self, task):
        """load neural activity signals input and label data from the .csv file
        
        task: behavior to decode
        """
        datasetName = os.path.join(self.root_dir, 'TRACES_'+ self.subject +'_'+ self.day +'_' + self.inj + '.csv')
        labelName = os.path.join(self.root_dir, 'BEHAVIOR_'+ self.subject +'_'+ self.day +'_' + self.inj + '.csv')
        
        X = np.loadtxt(datasetName, delimiter=',') # (N, K) (N: number of features(neurons), K: number of samples)
        X = np.transpose(X) # (K, N)
        
        y = np.loadtxt(labelName, delimiter=',')[:, task] # extract fine movement as labels (K, )

        # convert label type from float to int
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)  
        
        return {'input': X, 'label': y}


    def get_cls_bal(self, label):
        """calculate class balance for your dataset labels

        label: 
        """
        
        cls_bal = sum(label) / len(label)
        if cls_bal < 0.6 and cls_bal > 0.4:
            print("Balanced Class! - dataset {}/{}/{}: {:.3f}".format(self.subject, self.day, self.inj, cls_bal))
        else:
            print("ImBalanced Class! - dataset {}/{}/{}: {:.3f}".format(self.subject, self.day, self.inj, cls_bal))
        
        return cls_bal


    def handle_miss_val(self, input):
        """fill up the empty 1st feature of the raw input  with the average of other features for each time step

        input: 
        """
        input[:, 0] = input.mean(axis=1)

        return input


    def preprocess(self, data, win_size, model_type):
        """ preprocess data extract 2990 frames from each dataset, leaving out first 10 frames

        data: raw data
        window_size: previous time steps for input we want to consider for predicting each data point
        model: characteristics of neural decoding model
        """

        X, y = [], []

        # for recurent models,
        for i in range(data['input'].shape[0] - win_size): # 0 ~ 2989
            X.append(data['input'][i:(i + win_size), :]) # v = [10, 273] # 0:10 = frame 1 ~ 10
            y.append(data['label'][i + win_size]) # predict frame 11 ~ frame 3000 

        # Xs = [2990, window_size, 273] / ys = [2990, 1]
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        if model_type != 'recurrent':
            X = X.reshape(X.shape[0], -1) # Xs = [2990, 2730]

        return {'input': X, 'label': y}


if __name__ == '__main__':
    set1 = NeuralData('/Users/wonsmacbook/Google_Drive/github/Online_Learning_Neural_Decoding/dataset', '1004', '1', '4')
    data_blk1 = set1.load(2)
    print(data_blk1['input'].shape, data_blk1['label'].shape)
    data_blk1['input'] = set1.handle_miss_val(data_blk1['input'])
    data_blk1 = set1.preprocess(data_blk1, 10, 'recurrent')
    print(data_blk1['input'].shape, data_blk1['label'].shape)

