import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np 
from PIL import Image
from plot_save_tensor import *
from mean_std_value import MV_pixel, std_pixel

# 1: temperature, 2: salinity, 3: oxygen, 4: chlorophyll-a, 5: net primary production, 6: phosphate, 7: nitrate, 8: medium particulate carbon
target_variable = 3

path_directory = os.getcwd()
directory_tensor = path_directory + '/tensor/(12, 12, 20)/'

class SEADataset(Dataset):
    def __init__(self, path = None, transform = None):
        super().__init__()
      
        model_tensor = []
        directory_float = directory_tensor + 'model2015/'
        list_ptFIles = os.listdir(directory_float)
        for ptFiles in list_ptFIles:
            if ptFiles != 'datetime_2015.5.pt' and ptFiles != 'datetime_2015.49.pt' and ptFiles != 'datetime_2015.51.pt' and ptFiles != 'datetime_2015.52.pt':
                my_tensor = torch.load(directory_float + ptFiles)
                model_tensor.append(my_tensor[:, :, :-1, :, :])
        
        list_tensor = model_tensor

        number_channel = 8

        # normalization

        mean_value_pixel = MV_pixel(list_tensor)
        mean_tensor = torch.tensor(mean_value_pixel.reshape(1, number_channel, 1, 1, 1))
        std_value_pixel = std_pixel(list_tensor)
        std_tensor = torch.tensor(std_value_pixel.reshape(1, number_channel, 1, 1, 1))
        normalized_list = []
        for tensor in list_tensor:
            tensor = (tensor - mean_tensor) / std_tensor
            tensor[:, :, -1, :, :] = tensor[:, :, -2, :, :]
            tensor = tensor[:, :, :, :, 1:-1]
            tensor = tensor.float()
            normalized_list.append(tensor)
        self.list_files = normalized_list

        if path is not None:
            self.path = path
        else:
            raise Exception("Paths should be given as input to initialize the SEA class.")

    def __len__(self):
        return len(self.list_files) 

    def __getitem__(self, index):
        data = self.list_files[index]
        N, C, D, H, W = data.shape
        ds = data.reshape(C, D, H, W)
        X = ds[:3, :30, :30, :30]
        y = ds[target_variable, :30, :30, :30]
        y = torch.unsqueeze(y, 0)

        return X,y            
