# -*- coding: utf-8 -*-
"""utils.ipynb"""

import numpy as np
import matplotlib.pyplot as plt

def visualize_dist(input, max_value):
    """visualize the distribution of input features"""
    fig = plt.figure(figsize = (20,10))
    n_idx = np.arange(input.shape[1])
    n_value = input.mean(axis=0) / max_value

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("neuron index")
    ax.set_ylabel("mean neuron value") 
    # freq, bins, patches = ax.hist(wines['sulphates'], color='steelblue', bins=15, edgecolor='black', linewidth=1)
    ax.bar(n_idx, n_value, align='center', alpha=0.5)
    fig.show()


def visual_train(history):
    """visualize training results
    
    history: 
    """

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = np.arange(1, len(loss) + 1)

    fig = plt.figure(figsize = (12, 6))
    ax = fig.add_subplot(1, 2, 1)  
    ax.plot(epochs, loss, 'r', linestyle='--', label='Training loss')
    ax.plot(epochs, val_loss, 'b',linestyle='-',  label='Validation loss')
    ax.set_title('Training and validation loss', fontweight ="bold") 
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(loc='best', prop={'size': 15})

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(epochs, acc, 'r', linestyle='--', label='Training Accuracy')
    ax.plot(epochs, val_acc, 'b',linestyle='-',  label='Validation Accuracy')
    ax.set_title('Training and validation accuracy', fontweight ="bold") 
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.legend(loc='best', prop={'size': 15})

    plt.show()


def visual_pred(y_true, y_pred, data_point, method):
  """visualize ground truth and predicted neurons' fine movements

  y_true: ground truths of neurons' fine movements
  y_pred: predictions of neurons' fine movements
  """
  if method == 'lstm':
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
  
  time = np.arange(1, y_true.shape[0] + 1)

  plt.figure(figsize=(15,10))
  plt.plot(time, y_true, 'r', linestyle='-', label='fine movement_actual')
  plt.plot(time, y_pred, 'b', linestyle='-', label='fine movement_pred')
  plt.xlabel('time')
  plt.ylabel('y')
  plt.title('Training and validation accuracy for {}'.format(data_point))
  plt.legend()
  plt.show()