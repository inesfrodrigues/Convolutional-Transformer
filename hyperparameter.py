"""
hyperparameter for the implementation
"""

batch = 1
number_channel = 8 # 1: temperature, 2: salinity, 3: oxygen, 4: chlorophyll-a, 5: net primary production, 6: phosphate, 7: nitrate, 8: medium particulate carbon
latitude_interval = (36, 44)
longitude_interval = (2, 9)
depth_interval = (0, 600)
year_interval = (2016, 2017)
year = 2016
resolution = (12, 12, 20)

kindof = 'model2016'

if kindof == 'model2016':
    channels = [0, 1, 2, 3, 4, 5, 6, 7]