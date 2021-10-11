"""
model.py
"""
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.regularizers import l2
from keras.initializers import he_normal, he_uniform
from keras.layers import Input, Flatten, InputLayer, Dense, Dropout, LSTM, GRU
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import accuracy_score

class Model:
    def __init__(self, model, time_steps, num_features, units, lr_rate, drop_rate, opt):
        """define a Keras model
        
        model: model you want to build
        time_steps: number of steps in time-series data
        num_features: number of features in time-series data
        units: number of neurons in each layer
        lr_rate: learning rate for the optimizer
        drop_rate: dropout rate
        opt: function for the optimizer 
        """

        # inputs = Input(shape=(time_steps, num_features))
        self.timer = []
        self.acc = []

        self.model = Sequential()
        if model is 'gru':
            self.model.add(GRU(units, unroll=True, input_shape=(time_steps, num_features)))
        elif model is 'lstm':
            self.model.add(LSTM(units, unroll=True, input_shape=(time_steps, num_features)))
        self.model.add(Dropout(drop_rate)) # to regularize the dense layer
        self.model.add(Dense(1, activation='sigmoid'))

        if opt is 'adam':
            self.model.compile(optimizer=Adam(learning_rate=lr_rate), loss='binary_crossentropy', metrics = ['binary_accuracy'])
        if opt is 'sgd':
            self.model.compile(optimizer=SGD(learning_rate=lr_rate, nesterov=True), loss='binary_crossentropy', metrics = ['binary_accuracy'])


    def __str__(self):
        return f"<timer = {self.timer}, acc = {self.acc}, model = {self.model}>"


    def run(self):
        """ abstract method
        """
        pass


    def train(self, train_datasets, params, is_freeze):
        """pretrain model on different datasets

        train_datasets:
        params:
        is_freeze:
        """

        # stack datasets to build a new training set
        train_x = train_datasets[0]['input']
        train_y = train_datasets[0]['label']

        for i in range(len(train_datasets)-1):
            train_x = np.vstack((train_x, train_datasets[i+1]['input']))
            train_y = np.vstack((train_y, train_datasets[i+1]['label']))

        self.model.fit(train_x, train_y, epochs=params['epochs'], batch_size=params['batch'], verbose=0)

        # freeze all layers, but the last layer for fine-tuning
        if is_freeze:
            for layer in self.model.layers[:-1]:
                layer.trainable = False


class Online_Model(Model):
    def __init__(self, model, time_steps, num_features, units, lr_rate, drop_rate, opt):
        super().__init__(model, time_steps, num_features, units, lr_rate, drop_rate, opt)


    def run(self, params, train_datasets, test_datasets, is_freeze):
        # build models for the baseline and our method 
        
        super().train(train_datasets, params, is_freeze)

        # transfer online learning on a stream of datasets
        for idx, dataset in enumerate(test_datasets):            
            y_true = dataset['label']

            # calculate number of blocks(updates) for 50% of test set
            block_num = math.ceil(dataset['input'].shape[0] / params['p'])

            # for each block, make predictions and then update the model with the p frames in a block(update_time)
            start_time = time.time()
            for j in range(block_num):
                # make predictions and update on inputs
                if j == block_num-1:
                    # total num of data samples might make the last block not divisible by params['p']
                    y_pred_block = self.model.predict(dataset['input'][j*params['p']:])
                    y_pred_block = (y_pred_block > 0.5).astype('int32')

                    # online learning: update right away data block
                    self.model.fit(dataset['input'][j*params['p']:], dataset['label'][j*params['p']:], epochs=params['update_epoch'], verbose=0, batch_size=params['p'])
                else:
                    y_pred_block = self.model.predict(dataset['input'][j*params['p']:(j+1)*params['p']])
                    y_pred_block = (y_pred_block > 0.5).astype('int32')

                    # online learning: update right away data block
                    self.model.fit(dataset['input'][j*params['p']:(j+1)*params['p']], dataset['label'][j*params['p']:(j+1)*params['p']], epochs=params['update_epoch'], verbose=0, batch_size=params['p'])

                # append predictions for each block for evaluation in the end
                if j == 0: 
                    y_pred = y_pred_block
                else:
                    y_pred = np.append(y_pred, y_pred_block)
                
            # calculate prediction time per sample
            end_time = time.time()    
            dataset_time = (end_time - start_time) / dataset['input'].shape[0]

            # calculate metrics for performance evaluation
            self.timer.append(dataset_time)
            self.acc.append(accuracy_score(y_true, y_pred))
            

class Batch_Model(Model):
    def __init__(self, model, time_steps, num_features, units, lr_rate, drop_rate, opt):
        super().__init__(model, time_steps, num_features, units, lr_rate, drop_rate, opt)


    def run(self, params, train_datasets, test_datasets, is_transfer, is_freeze):
        """batch train and evaluate the model

        params:
        train_datasets:
        test_datasets:
        is_transfer:
        """
        
        super().train(train_datasets, params, is_freeze)

        # make predictions on a stream of test datasets and fine-tune models on the training set
        for idx, dataset in enumerate(test_datasets):
            y_true = dataset['label']

            # make predictions on the current set of features
            start_time = time.time()
            y_pred = self.model.predict(dataset['input'])
            y_pred = (y_pred > 0.5).astype('int32')
            end_time = time.time()
            dataset_time = (end_time - start_time) / dataset['input'].shape[0]

            # calculate metrics for performance evaluation
            self.timer.append(dataset_time)
            self.acc.append(accuracy_score(y_true, y_pred))


if __name__ == '__main__':
    a = Batch_Model(model='lstm', time_steps=10, num_features=273, units=30, lr_rate=0.01, drop_rate=0.25, opt='adam')
    b = Online_Model(model='lstm', time_steps=10, num_features=273, units=30, lr_rate=0.01, drop_rate=0.25, opt='adam')