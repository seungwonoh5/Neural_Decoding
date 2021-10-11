"""
data.py
"""
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

class NeuralData:
    def __init__(self, root_dir, subject, day, inj, task):
        """ load neural data 
        conda install -c conda-forge scikit-learn 
        root_dir: str, path to the csv file (data and annotations)
        subject: list[tuple], list of tuples that each represents a dataset with a specific day and injection condition
        day: int, the number of bins you want to use together to predict y
        inj: int, the 
        task: int, the behavior variable that we want to predict
        """
        self.datasetName = os.path.join(root_dir, 'TRACES_'+ subject +'_'+ day +'_' + inj + '.csv')
        self.labelName = os.path.join(root_dir, 'BEHAVIOR_'+ subject +'_'+ day +'_' + inj + '.csv')
        # (number of features/neurons, number of samples)
        self.X = np.transpose(np.loadtxt(self.datasetName, delimiter=',')) 

        # filling missing values in the csv
        self.X[:, 0] = self.X.mean(axis=1) 

        # convert label type from float to int
        self.y = np.loadtxt(self.labelName, delimiter=',')[:, task]
        encoder = LabelEncoder()
        encoder.fit(self.y)
        self.y = encoder.transform(self.y) 

    
    def __str__(self):
        return f"<datasetName = {self.datasetName}, labelName = {self.labelName}, X = {self.X.shape}, y = {self.y.shape}>"
    

    def get_stats(self):
        return sum(self.y) / len(self.y)


    def preprocess(self, win_size, time_series=True):
        """ preprocess data into real-time neural decoding task

        win_size: previous time steps for input we want to consider for predicting each data point
        time_series: whether the data will be converted into a time-series
        """

        X, y = [], []

        # for recurrent models,
        for i in range(self.X.shape[0] - win_size): # 0 ~ 2989
            X.append(self.X[i:(i + win_size), :]) # v = [10, 273] # 0:10 = frame 1 ~ 10
            y.append(self.y[i + win_size]) # predict frame 11 ~ frame 3000 

        # Xs = [2990, window_size, 273] / ys = [2990, 1]
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        if time_series == False:
            X = X.reshape(X.shape[0], -1) # Xs = [2990, 2730]

        return {'input': X, 'label': y}

if __name__ == '__main__':
    set1 = NeuralData('data', '1004', '1', '4', 2)
    pset1 = set1.preprocess(50, True)
    print(pset1['input'].shape, pset1['label'].shape)