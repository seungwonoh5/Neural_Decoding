"""
utils.py
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_dist(inp, max_value):
    """visualize the distribution of input features
    
    input:
    max_value:
    """
    fig = plt.figure(figsize = (20,10))
    n_idx = np.arange(inp.shape[1])

    # normalize values to max_value to make it b/w 0 and 1
    n_value = inp.mean(axis=0) / max_value

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("neuron id")
    ax.set_ylabel("avg neuron value") 
    ax.bar(n_idx, n_value, align='center', alpha=0.5)
    fig.show()


def plot_performance(batch_perform, online_perform):
    """visualize ground truth and predicted neurons' fine movements
    
    batch_perform: list, batch metric per dataset
    online_perform: list, online metric per dataset
    """

    # plt.figure(figsize=(15,10))
    x = np.arange(len(batch_perform)) + 1
    plt.plot(x, batch_perform, 'r', linestyle='-', label='batch')
    plt.plot(x, online_perform, 'b', linestyle='-', label='online')
    plt.xlabel('dataset')
    plt.ylabel('accuracy')
    plt.title('[test accuracy] online vs. batch')
    plt.legend()
    plt.show()