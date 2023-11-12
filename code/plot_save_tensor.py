"""
Plotting the information contained in the tensors
"""
import matplotlib.pyplot as plt
import os
import torch
from hyperparameter2 import *

path = "fig/"
path_directory = os.getcwd()


def Plot_Tensor(kindof, tensor, data_time, channel, flag):
    """
    Plotting the tensor's values at different levels of depth
    i.e. plotting along the component d (=depth) of tensor (bs, c,c d, h, w)
    tensor = tensor we want to plot (i.e. parallelepiped at a fixed date time)
    data_time = reference date time associated to the list of tensor
    channel = variable we want to plot
    """
    dict_channel = {0: 'temperature', 1: 'salinity', 2: 'oxygen', 3: 'chla', 4: 'npp', 5: 'n1p', 6: 'n3n', 7: 'r6c'}
    if flag == 'w':
        directory = path_directory + '/weight/' + str(kindof) + '/' + str(channel) + '/' + str(data_time)
    if flag == 't':
        directory = path_directory + '/fig/' + str(kindof) + '/' + str(channel) + '/' + str(data_time)

    if not os.path.exists(directory):
        os.mkdir(directory)

    number_fig = len(tensor[0, 0, :, 0, 0])  # number of levels of depth

    for i in range(number_fig):
        cmap = plt.get_cmap('Greens')
        plt.imshow(tensor[0, channel, i, :, :], cmap=cmap)
        if flag == 'w':
            plt.title(dict_channel[channel] + 'weight')
        if flag == 't':
            plt.title(dict_channel[channel])
        plt.colorbar()
        plt.savefig(directory + "/profondity_level_" + str(i) + ".png")
        plt.close()


def plot_routine(kindof, list_parallelepiped, list_data_time, channels, year_interval, flag):
    """
    measurement plot different for each kind of data (float/sat/tensor)
    kindof = requires a str (float, sat or tensor)
    list_parallelepiped = list of tensor we want to plot (i.e. parallelepiped at a fixed date time)
    list_data_time = list of reference date time associated to the list of tensor
    channels = list of variable we want to plot
    flag = we are plotting the tensor or the weight, if no specify the tensor
    """
    year_min, year_max = year_interval
    for j in range(len(list_data_time)):
        print('ola')
        time_considered = list_data_time[j]
        tensor_considered = list_parallelepiped[j]
        if year_min < time_considered < year_max:
            print('plotting tensor relative to time : ', time_considered)
            for channel in channels:
                Plot_Tensor(kindof, tensor_considered, time_considered, channel, flag)


def Save_Tensor(kindof, tensor, data_time, flag):
    """
    Saving the tensor's values at different levels of depth
    i.e. plotting along the component d (=depth) of tensor (bs, c,c d, h, w)
    tensor = tensor we want to plot (i.e. parallelepiped at a fixed date time)
    data_time = reference date time associated to the list of tensor
    channel = variable we want to plot
    """
    if flag == 'w':
        directory = path_directory + '/weight_tensor/'
    else:
        directory = path_directory + '/tensor/'
    specific_directory = directory + str(resolution) + '/'
    if not os.path.exists(specific_directory):
        os.mkdir(specific_directory)
    final_directory = specific_directory + str(kindof)
    if not os.path.exists(final_directory):
        os.mkdir(final_directory)

    torch.save(tensor, final_directory + "/datetime_" + str(data_time) + ".pt")


def save_routine(kindof, list_parallelepiped, list_data_time, year_interval, flag):
    """
    measurement plot different for each kind of data (float/sat/tensor)
    kindof = requires a str (float, sat or tensor)
    list_parallelepiped = list of tensor we want to plot (i.e. parallelepiped at a fixed date time)
    list_data_time = list of reference date time associated to the list of tensor
    channels = list of variable we want to plot
    flag = we are plotting the tensor or the weight, if no specify the tensor
    """
    year_min, year_max = year_interval
    for j in range(len(list_data_time)):
        print('ola')
        time_considered = list_data_time[j]
        tensor_considered = list_parallelepiped[j]
        if year_min < time_considered < year_max:
            print('saving tensor relative to time : ', time_considered)
            Save_Tensor(kindof, tensor_considered, time_considered, flag)
