"""
Function that compute the mean/std value of the pixels of the training set
I : train_dataset i.e. a list of 4 channel tensor
O : channel_total_mean i.e. a numpy array containing the mean along the channel values of the input training set
Implementation for a problem with 4 channel to estimate
"""
from hyperparameter import *
import numpy as np


def MV_pixel(train_dataset):
    channel_total_mean = np.zeros(shape=(number_channel,))
    for train_tensor in train_dataset:
        tensor_mean = np.array(train_tensor.mean(axis=(0, 2, 3, 4)))  # mean value of the different channel
        channel_total_mean = channel_total_mean + tensor_mean
    channel_total_mean = channel_total_mean / len(train_dataset)
    return channel_total_mean


def std_pixel(train_dataset):
    channel_total_std = np.zeros(shape=(number_channel,))
    for train_tensor in train_dataset:
        tensor_std = np.array(train_tensor.std(axis=(0, 2, 3, 4)))  # mean value of the different channel
        channel_total_std = channel_total_std + tensor_std
    channel_total_std = channel_total_std / len(train_dataset)
    return channel_total_std
