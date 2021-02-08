# -*- coding: utf-8 -*-
"""models.ipynb"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.regularizers import l2
from keras.initializers import he_normal, he_uniform
from keras.layers import Input, InputLayer, Dense, Dropout, LSTM, GRU
from keras.optimizers import Adam, SGD
from keras.metrics import BinaryAccuracy

# non-recurrent models
def mlp(n_layers, input_dim, units, lr_rate, drop_rate, seed, opt):
    """define a model stacked LSTM layers

    n_layers: number of LSTM layers in the architecture
    units: number of neurons in each LSTM layer
    lr_rate: learning rate for the optimizer
    drop_rate:
    seed:
    opt:
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(input_dim,)))
    for _ in range(n_layers):
        model.add(Dense(units, kernel_initializer = he_uniform(seed=seed), activation='relu'))
        model.add(Dropout(rate=drop_rate, seed=seed))
    model.add(Dense(1, activation='sigmoid'))

    if opt == 'adam':
        model.compile(optimizer=Adam(lr=lr_rate), loss='binary_crossentropy', metrics = ['binary_accuracy'])
    if opt == 'sgd':
        model.compile(optimizer=SGD(lr=lr_rate, nesterov=True), loss='binary_crossentropy', metrics = ['binary_accuracy'])

    model.summary()
    return model


# recurrent models
def gru(n_layers, input_dims, units, lr_rate, r_drop_rate, drop_rate, opt):
    """define a model stacked GRU layers

    n_layers: number of LSTM layers in the architecture
    units: number of neurons in each LSTM layer
    lr_rate: learning rate for the optimizer
    r_drop_rate: dropout rate for recurrent cells
    drop_rate: dropout rate for cells
    opt: optimizer function
    """
    model = Sequential()
    model.add(GRU(units, return_sequences=True, recurrent_dropout=r_drop_rate, dropout=drop_rate, input_shape=(input_dims.shape[1], input_dims.shape[2])))
    for _ in range(n_layers):
        model.add(GRU(units, return_sequences=True, recurrent_dropout=r_drop_rate, dropout=drop_rate))
    model.add(GRU(units, recurrent_dropout=r_drop_rate, dropout=drop_rate))
    model.add(Dense(1, activation='sigmoid'))

    if opt is 'adam':
        model.compile(optimizer=Adam(lr=lr_rate), loss='binary_crossentropy', metrics = ['binary_accuracy'])
    if opt is 'sgd':
        model.compile(optimizer=SGD(lr=lr_rate, nesterov=True), loss='binary_crossentropy', metrics = ['binary_accuracy'])

    model.summary()
    return model


def lstm(n_layers, input_dims, units, lr_rate, r_drop_rate, drop_rate, opt):
    """define a model stacked LSTM layers

    n_layers: number of LSTM layers in the architecture
    units: number of neurons in each LSTM layer
    lr_rate: learning rate for the optimizer
    r_drop_rate:
    drop_rate:
    opt:
    """
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, recurrent_dropout=r_drop_rate, dropout=drop_rate, input_shape=(input_dims.shape[1], input_dims.shape[2])))
    for _ in range(n_layers):
        model.add(LSTM(units, return_sequences=True, recurrent_dropout=r_drop_rate, dropout=drop_rate))
    model.add(LSTM(units, recurrent_dropout=r_drop_rate, dropout=drop_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    if opt == 'adam':
        model.compile(optimizer=Adam(lr=lr_rate), loss='binary_crossentropy', metrics = ['binary_accuracy'])
    if opt == 'sgd':
        model.compile(optimizer=SGD(lr=lr_rate, nesterov=True), loss='binary_crossentropy', metrics = ['binary_accuracy'])
    
    model.summary()
    return model


if __name__ == '__main__':
    a = mlp(n_layers=3, input_dim=273, units=30, lr_rate=0.01, drop_rate=0.25, seed=42, opt='adam')
