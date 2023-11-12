"""
Plotting the error during the different phase of the training
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np


def Plot_Error(losses, flag, path):
    """
    plot of the losses
    flag : what phase of the training we are plotting
    path : where to save the plot
    """
    flag_dict = {'test': 'test dataset', 'train': 'train dataset'}
    flag_color = {'test': 'blue', 'train': 'orange'}

    descr = flag_dict[flag]
    color = flag_color[flag]

    figure(figsize=(10, 6))

    label = 'losses'
    #losses = losses.detach().numpy()
    plt.plot(losses, color, label=label)
    plt.xlabel('Number of epochs')
    plt.title('Losses of ' + descr)
    plt.legend()
    plt.savefig(path + "loss_" + str(flag) + ".png")
    plt.close()

    figure(figsize=(10, 6))

    label = 'log losses '
    plt.plot(np.log10(losses), color, label=label)
    plt.xlabel('Number of epochs')
    plt.title('Logarithmic Losses of ' + descr)
    plt.legend()
    plt.savefig(path + "LOGloss_" + str(flag) + ".png")
    plt.close()


def Plot_Error_Both(losses, losses2, path):

    figure(figsize=(10, 6))

    label = 'train'
    label2 = 'test'
    plt.plot(losses, 'orange', label=label)
    plt.plot(losses2, 'blue', label=label2)
    plt.xlabel('Number of epochs')
    plt.title('Losses of Train and Test Datasets')
    plt.legend()
    plt.savefig(path + "loss_train_test" + ".png")
    plt.close()

    figure(figsize=(10, 6))

    label = 'train'
    label2 = 'test'
    plt.plot(np.log10(losses), 'orange', label=label)
    plt.plot(np.log10(losses2), 'blue', label=label2)
    plt.xlabel('Number of epochs')
    plt.title('Logarithmic Losses of Train and Test Datasets')
    plt.legend()
    plt.savefig(path + "LOGloss_train_test" + ".png")
    plt.close()