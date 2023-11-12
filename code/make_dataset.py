"""
Routine for the creation of the parallelepiped that composed the training set
"""
import torch
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
from plot_save_tensor import plot_routine, save_routine
from hyperparameter import *

constant_latitude = 111 # 1° of latitude corresponds to 111 km
constant_longitude = 111 # 1° of latitude corresponds to 111 km
float_path = "../FLOAT_BIO/"


def read_date_time_sat(date_time):
  """
  Take as input a date-time in str format and decode it in a format considering only year and month
  year + 0.01 * week
  """
  year = int(date_time[0:4])
  month = int(date_time[4:6])
  month = month - 1
  day = int(date_time[6:8])
  week = int(month * 4 + day / 7)
  date_time_decoded = year + 0.01 * week
  return date_time_decoded


def to_depth(press, latitude):
  """
  convert press input in depth one
  press = pressure in decibars
  lat = latitude in deg
  depth = depth in metres
  """
  x = np.sin(latitude / 57.29578)
  x = x * x
  gr = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x) + 1.092e-6 * press
  depth = (((-1.82e-15 * press + 2.279e-10) * press - 2.2512e-5) * press + 9.72659) * press / gr
  return depth


def create_list_date_time(years_consider):
  """
  Creation of a list containing date_time reference for training dataset
  years_consider = (first year considered, last year considered)
  interval_of_time = intervals within measurement are aggregated
  """
  first_year_considered, last_year_considered = years_consider
  total_list = []
  for year in np.arange(first_year_considered, last_year_considered):
    lists = np.arange(year, year + 0.53, 0.01)
    for i in range(len(lists)):
      lists[i] = round(lists[i], 2)
    lists = lists.tolist()
    total_list = total_list + lists
  return total_list


def create_box(batch, number_channel, lat, lon, depth, resolution):
  """
  Function that creates the EMPTY tensor that will be filled with data
  batch = batch size/ batch number ?
  number_channel = number of channel (i.e. unknowns we want to predict)
  lat = (lat_min, lat_max)
  lon = (lon_min, lon_max)
  depth = (depth_min, depth_max) in km
  resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
  output = tensor zeros (MB, C, D, H, W)
  """
  lat_min, lat_max = lat
  lon_min, lon_max = lon
  depth_min, depth_max = depth
  w_res, h_res, d_res = resolution
  w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
  h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
  d = int((depth_max - depth_min) / d_res + 1)
  empty_parallelepiped = torch.zeros(batch, number_channel, d, h, w)
  return empty_parallelepiped


def find_index(lat, lat_limits, lat_size):
  """
  Function that given a latitude/longitude/depth as input return the index where to place it in the tensor
  lat = latitude considered
  lat_limits = (lat_min, lat_max)
  lat_size = dimension of latitude dmensin in the tensor
  """
  lat_min, lat_max = lat_limits
  lat_res = (lat_max - lat_min) / lat_size
  lat_index = int((lat - lat_min) / lat_res)
  return lat_index

def insert_model_temp_values(year, lat_limits, lon_limits, depth_limits, year_limits, resolution):
  """
    function that update the parallelepiped updating all the voxel with MODEL information
    year = folder of the year we are considering
    lat_limits = (lat_min, lat_max)
    lon_limits = (lon_min, lon_max)
    depth_limits = (depth_min, depth_max) in km
    year_limits = (year_min, year_max)
    resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
  """
  lat_min, lat_max = lat_limits
  lon_min, lon_max = lon_limits
  depth_min, depth_max = depth_limits
  year_min, year_max = year_limits
  w_res, h_res, d_res = resolution

  w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
  h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
  d = int((depth_max - depth_min) / d_res + 1)

  path_doxy = os.getcwd() + "/dataset/MODEL/" + str(year) + '/votemper/'
  doxy_files = os.listdir(path_doxy)
  for model_file in doxy_files:
    if model_file[0:3] != 'ave':
      continue
    file_doxy = path_doxy + model_file
    ds_doxy = nc.Dataset(file_doxy)

    time = model_file[4:12]
    time = read_date_time_sat(time)
    if not year_min < time < year_max:
      continue
    index = list_data_time.index(time) # index input tens considered, i.e. the one to upd
    select_parallelepiped = list_parallelepiped[index] # parall we are modifying

    latitude_list = ds_doxy['lat'][:].data
    longitude_list = ds_doxy['lon'][:].data
    depth_list = ds_doxy['depth'][:].data

    doxy_tens = torch.tensor(ds_doxy['votemper'][:].data)[0, :, :, :] # tensor indexes as temp(depth, x, y)

    print(model_file + ' analysis started')
    for i in range(len(latitude_list)): # indexing over the latitude (3rd component of the tensor)
      print('temp')
      for j in range(len(longitude_list)): # indexing over the longitude (2nd component of the tensor)
        for k in range(len(depth_list)): # indexing over the depth (1st component of the tensor)
          latitude = latitude_list[i]
          longitude = longitude_list[j]
          depth = depth_list[k]

          doxy = float(doxy_tens[k, i, j].item())

          if lat_max > latitude > lat_min:
            if lon_max > longitude > lon_min:
              if depth_max > depth > depth_min:
                latitude_index = find_index(latitude, lat_limits, w)
                longitude_index = find_index(longitude, lon_limits, h)
                depth_index = find_index(depth, depth_limits, d)

                if -3 < doxy < 40:
                  select_parallelepiped[0, 0, depth_index, longitude_index, latitude_index] = doxy
    print(model_file + ' analysis completed')
    np.save(os.getcwd() + "/saved_parallelepiped/temp_parallelepiped2016.npy", list_parallelepiped)
  return

def insert_model_sal_values(year, lat_limits, lon_limits, depth_limits, year_limits, resolution):
  """
    function that update the parallelepiped updating all the voxel with MODEL information
    year = folder of the year we are considering
    lat_limits = (lat_min, lat_max)
    lon_limits = (lon_min, lon_max)
    depth_limits = (depth_min, depth_max) in km
    year_limits = (year_min, year_max)
    resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
  """
  lat_min, lat_max = lat_limits
  lon_min, lon_max = lon_limits
  depth_min, depth_max = depth_limits
  year_min, year_max = year_limits
  w_res, h_res, d_res = resolution

  w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
  h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
  d = int((depth_max - depth_min) / d_res + 1)

  path_doxy = os.getcwd() + "/dataset/MODEL/" + str(year) + '/vosaline/'
  doxy_files = os.listdir(path_doxy)
  for model_file in doxy_files:
    if model_file[0:3] != 'ave':
      continue
    file_doxy = path_doxy + model_file
    ds_doxy = nc.Dataset(file_doxy)

    time = model_file[4:12]
    time = read_date_time_sat(time)
    if not year_min < time < year_max:
      continue
    index = list_data_time.index(time) # index input tens considered, i.e. the one to upd
    select_parallelepiped = list_parallelepiped[index] # parall we are modifying

    latitude_list = ds_doxy['lat'][:].data
    longitude_list = ds_doxy['lon'][:].data
    depth_list = ds_doxy['depth'][:].data

    doxy_tens = torch.tensor(ds_doxy['vosaline'][:].data)[0, :, :, :] # tensor indexes as temp(depth, x, y)

    print(model_file + ' analysis started')
    for i in range(len(latitude_list)): # indexing over the latitude (3rd component of the tensor)
      print('sal')
      for j in range(len(longitude_list)): # indexing over the longitude (2nd component of the tensor)
        for k in range(len(depth_list)): # indexing over the depth (1st component of the tensor)
          latitude = latitude_list[i]
          longitude = longitude_list[j]
          depth = depth_list[k]

          doxy = float(doxy_tens[k, i, j].item())

          if lat_max > latitude > lat_min:
            if lon_max > longitude > lon_min:
              if depth_max > depth > depth_min:
                latitude_index = find_index(latitude, lat_limits, w)
                longitude_index = find_index(longitude, lon_limits, h)
                depth_index = find_index(depth, depth_limits, d)

                if 2 < doxy < 41:
                  select_parallelepiped[0, 1, depth_index, longitude_index, latitude_index] = doxy
    print(model_file + ' analysis completed')
    np.save(os.getcwd() + "/saved_parallelepiped/sal_parallelepiped2016.npy", list_parallelepiped)
  return


def insert_model_doxy_values(year, lat_limits, lon_limits, depth_limits, year_limits, resolution):
  """
    function that update the parallelepiped updating all the voxel with MODEL information
    year = folder of the year we are considering
    lat_limits = (lat_min, lat_max)
    lon_limits = (lon_min, lon_max)
    depth_limits = (depth_min, depth_max) in km
    year_limits = (year_min, year_max)
    resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
  """
  lat_min, lat_max = lat_limits
  lon_min, lon_max = lon_limits
  depth_min, depth_max = depth_limits
  year_min, year_max = year_limits
  w_res, h_res, d_res = resolution

  w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
  h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
  d = int((depth_max - depth_min) / d_res + 1)

  path_doxy = os.getcwd() + "/dataset/MODEL/" + str(year) + '/O2o/'
  doxy_files = os.listdir(path_doxy)
  for model_file in doxy_files:
    if model_file[0:3] != 'ave':
      continue
    file_doxy = path_doxy + model_file
    ds_doxy = nc.Dataset(file_doxy)

    time = model_file[4:12]
    time = read_date_time_sat(time)
    if not year_min < time < year_max:
      continue
    index = list_data_time.index(time) # index input tens considered, i.e. the one to upd
    select_parallelepiped = list_parallelepiped[index] # parall we are modifying

    latitude_list = ds_doxy['lat'][:].data
    longitude_list = ds_doxy['lon'][:].data
    depth_list = ds_doxy['depth'][:].data

    doxy_tens = torch.tensor(ds_doxy['O2o'][:].data)[0, :, :, :] # tensor indexes as temp(depth, x, y)

    print(model_file + ' analysis started')
    for i in range(len(latitude_list)): # indexing over the latitude (3rd component of the tensor)
      print('doxy')
      for j in range(len(longitude_list)): # indexing over the longitude (2nd component of the tensor)
        for k in range(len(depth_list)): # indexing over the depth (1st component of the tensor)
          latitude = latitude_list[i]
          longitude = longitude_list[j]
          depth = depth_list[k]

          doxy = float(doxy_tens[k, i, j].item())

          if lat_max > latitude > lat_min:
            if lon_max > longitude > lon_min:
              if depth_max > depth > depth_min:
                latitude_index = find_index(latitude, lat_limits, w)
                longitude_index = find_index(longitude, lon_limits, h)
                depth_index = find_index(depth, depth_limits, d)

                if -5 < doxy < 600:
                  select_parallelepiped[0, 2, depth_index, longitude_index, latitude_index] = doxy
    print(model_file + ' analysis completed')
    np.save(os.getcwd() + "/saved_parallelepiped/doxy_parallelepiped2016.npy", list_parallelepiped)
  return


def insert_model_chl_values(year, lat_limits, lon_limits, depth_limits, year_limits, resolution):
  """
    function that update the parallelepiped updating all the voxel with MODEL information
    year = folder of the year we are considering
    lat_limits = (lat_min, lat_max)
    lon_limits = (lon_min, lon_max)
    depth_limits = (depth_min, depth_max) in km
    year_limits = (year_min, year_max)
    resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
  """
  lat_min, lat_max = lat_limits
  lon_min, lon_max = lon_limits
  depth_min, depth_max = depth_limits
  year_min, year_max = year_limits
  w_res, h_res, d_res = resolution

  w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
  h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
  d = int((depth_max - depth_min) / d_res + 1)

  path_chl = os.getcwd() + "/dataset/MODEL/" + str(year) + '/P_l/'
  chl_files = os.listdir(path_chl)
  for model_file in chl_files:
    if model_file[0:3] != 'ave':
      continue
    file_chl = path_chl + model_file
    ds_chl = nc.Dataset(file_chl)

    time = model_file[4:12]
    time = read_date_time_sat(time)
    if not year_min < time < year_max:
      continue
    index = list_data_time.index(time) # index input tens considered, i.e. the one to upd
    select_parallelepiped = list_parallelepiped[index] # parall we are modifying

    latitude_list = ds_chl['lat'][:].data
    longitude_list = ds_chl['lon'][:].data
    depth_list = ds_chl['depth'][:].data

    chl_tens = torch.tensor(ds_chl['P_l'][:].data)[0, :, :, :] # tensor indexes as temp(depth, x, y)

    print(model_file + ' analysis started')
    for i in range(len(latitude_list)): # indexing over the latitude (3rd component of the tensor)
      print('chl')
      for j in range(len(longitude_list)): # indexing over the longitude (2nd component of the tensor)
        for k in range(len(depth_list)): # indexing over the depth (1st component of the tensor)
          latitude = latitude_list[i]
          longitude = longitude_list[j]
          depth = depth_list[k]

          chl = float(chl_tens[k, i, j].item())

          if lat_max > latitude > lat_min:
            if lon_max > longitude > lon_min:
              if depth_max > depth > depth_min:
                latitude_index = find_index(latitude, lat_limits, w)
                longitude_index = find_index(longitude, lon_limits, h)
                depth_index = find_index(depth, depth_limits, d)

                if -5 < chl < 600:
                  select_parallelepiped[0, 3, depth_index, longitude_index, latitude_index] = chl
    print(model_file + ' analysis completed')
    np.save(os.getcwd() + "/saved_parallelepiped/chl_parallelepiped2016.npy", list_parallelepiped)

  return


def insert_model_ppn_values(year, lat_limits, lon_limits, depth_limits, year_limits, resolution):
  """
    function that update the parallelepiped updating all the voxel with MODEL information
    year = folder of the year we are considering
    lat_limits = (lat_min, lat_max)
    lon_limits = (lon_min, lon_max)
    depth_limits = (depth_min, depth_max) in km
    year_limits = (year_min, year_max)
    resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
  """
  lat_min, lat_max = lat_limits
  lon_min, lon_max = lon_limits
  depth_min, depth_max = depth_limits
  year_min, year_max = year_limits
  w_res, h_res, d_res = resolution

  w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
  h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
  d = int((depth_max - depth_min) / d_res + 1)

  path_chl = os.getcwd() + "/dataset/MODEL/" + str(year) + '/ppn/'
  chl_files = os.listdir(path_chl)
  for model_file in chl_files:
    if model_file[0:3] != 'ave':
      continue
    file_chl = path_chl + model_file
    ds_chl = nc.Dataset(file_chl)

    time = model_file[4:12]
    time = read_date_time_sat(time)
    if not year_min < time < year_max:
      continue
    index = list_data_time.index(time) # index input tens considered, i.e. the one to upd
    select_parallelepiped = list_parallelepiped[index] # parall we are modifying

    latitude_list = ds_chl['lat'][:].data
    longitude_list = ds_chl['lon'][:].data
    depth_list = ds_chl['depth'][:].data

    chl_tens = torch.tensor(ds_chl['ppn'][:].data)[0, :, :, :] # tensor indexes as temp(depth, x, y)

    print(model_file + ' analysis started')
    for i in range(len(latitude_list)): # indexing over the latitude (3rd component of the tensor)
      print('ppn')
      for j in range(len(longitude_list)): # indexing over the longitude (2nd component of the tensor)
        for k in range(len(depth_list)): # indexing over the depth (1st component of the tensor)
          latitude = latitude_list[i]
          longitude = longitude_list[j]
          depth = depth_list[k]

          chl = float(chl_tens[k, i, j].item())

          if lat_max > latitude > lat_min:
            if lon_max > longitude > lon_min:
              if depth_max > depth > depth_min:
                latitude_index = find_index(latitude, lat_limits, w)
                longitude_index = find_index(longitude, lon_limits, h)
                depth_index = find_index(depth, depth_limits, d)

                if -5 < chl < 600:
                  select_parallelepiped[0, 4, depth_index, longitude_index, latitude_index] = chl
    print(model_file + ' analysis completed')
    np.save(os.getcwd() + "/saved_parallelepiped/ppn_parallelepiped2016.npy", list_parallelepiped)

  return


def insert_sat_values(lat_limits, lon_limits, depth_limits, year_limits, resolution):
  """
  function that update the parallelepiped updating the voxel on the surfaces
  the only information provided is the 'CHL' ones
  lat_limits = (lat_min, lat_max)
  lon_limits = (lon_min, lon_max)
  depth_limits = (depth_min, depth_max) in km
  resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
  """
  lat_min, lat_max = lat_limits
  lon_min, lon_max = lon_limits
  depth_min, depth_max = depth_limits
  year_min, year_max = year_limits
  w_res, h_res, d_res = resolution

  w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
  h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
  d = int((depth_max - depth_min) / d_res + 1)

  path_sat = os.getcwd() + "/dataset/WEEKLY_1_24/"
  sat_measurement = os.listdir(path_sat)

  for sat_file in sat_measurement:
    file = path_sat + sat_file
    ds = nc.Dataset(file)

    datatime = sat_file[0:8]
    time = read_date_time_sat(datatime)
    if not year_min < time < year_max:
      print('time out of range', time)
      continue
    index = list_data_time.index(time) # index input tens considered, i.e. the one to upd
    select_parallelepiped = list_parallelepiped[index] # parall we are modifying
    select_weigth = list_weight_sat[index] # weight tensor we are modifying

    latitude_list = ds['lat'][:].data
    longitude_list = ds['lon'][:].data

    depth = float(ds['depth'][:].data)
    depth_index = find_index(depth, depth_limits, d) # 0 bc we are on the surfaces

    matrix_chl = ds['CHL'][0::].data[0]

    for i in range(len(latitude_list)):
      for j in range(len(longitude_list)):
        lat = latitude_list[i]
        lon = longitude_list[j]
        chl = matrix_chl[i, j]
        if -5 < chl < 600:
          if lat_max > lat > lat_min:
            if lon_max > lon > lon_min:
              lat_index = find_index(lat, lat_limits, w)
              lon_index = find_index(lon, lon_limits, h)
              select_parallelepiped[0, 3, depth_index, lon_index, lat_index] = float(chl)
              select_weigth[0, 3, depth_index, lon_index, lat_index] = 1.0
  return


def insert_float_values(lat_limits, lon_limits, depth_limits, year_limits, resolution):
  """
  Function that update the parallelepiped updating the voxel where the float info is available
  lat_limits = (lat_min, lat_max)
  lon_limits = (lon_min, lon_max)
  depth_limits = (depth_min, depth_max) in km
  year_limits = (year_min, year_max)
  resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
  """
  lat_min, lat_max = lat_limits
  lon_min, lon_max = lon_limits
  depth_min, depth_max = depth_limits
  year_min, year_max = year_limits
  w_res, h_res, d_res = resolution

  w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
  h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
  d = int((depth_max - depth_min) / d_res + 1)

  list_data = pd.read_csv(float_path + 'data/Float_Index.txt', header=None).to_numpy()[:, 0].tolist()
  list_datetime = pd.read_csv(float_path + 'data/Float_Index.txt', header=None).to_numpy()[:, 3].tolist()

  for i in range(np.size(list_data)): # indexing on list_data and list_datetime also
    path_current_float = float_path + "data/" + list_data[i]
    ds = nc.Dataset(path_current_float)

    var_list = []
    for var in ds.variables:
      var_list.append(var)

    datetime = list_datetime[i]
    time = read_date_time_sat(datetime)
    if not year_min < time < year_max:
      print('time out of range', time)
      continue

    index = list_data_time.index(time) # index input tens considered, i.e. the one to upd
    select_parallelepiped = list_parallelepiped[index] # parall we are modifying
    select_weigth = list_weight_float[index] # weight tensor we are modifying

    lat = float(ds['LATITUDE'][:].data) # single value
    lon = float(ds['LONGITUDE'][:].data) # single value

    lat_index = find_index(lat, lat_limits, w)
    lon_index = find_index(lon, lon_limits, h)

    pres_list = ds['PRES'][:].data[0] # list of value
    depth_list = []
    for pres in pres_list:
      depth_list.append(to_depth(pres, lat))

    temp = ds['TEMP'][:].data[0] # list of value
    salinity = ds['PSAL'][:].data[0]
    if 'DOXY' in var_list:
      doxy = ds['DOXY'][:].data[0]
    if 'CHLA' in var_list:
      chla = ds['CHLA'][:].data[0]

    if lat_max > lat > lat_min:
      if lon_max > lon > lon_min:
        for depth in depth_list:
          if depth_max > depth > depth_min:
            depth_index = find_index(depth, depth_limits, d)
            channel_index = np.where(depth_list == depth)[0][0]

            temp_v, salinity_v = temp[channel_index], salinity[channel_index]

            if not -3 < temp_v < 40:
              print('invalid temperature found', temp_v)
            else:
              select_parallelepiped[0, 0, depth_index, lon_index, lat_index] = float(temp_v)
              select_weigth[0, 0, depth_index, lon_index, lat_index] = 1.0

            if not 2 < salinity_v < 41:
              print('invalid psal found', salinity_v)
            else:
              select_parallelepiped[0, 1, depth_index, lon_index, lat_index] = float(salinity_v)
              select_weigth[0, 1, depth_index, lon_index, lat_index] = 1.0

            if 'DOXY' in var_list:
              doxy_v = doxy[channel_index]
              if not -5 < doxy_v < 600:
                print('invalid doxy found', doxy_v)
              else:
                select_parallelepiped[0, 2, depth_index, lon_index, lat_index] = float(doxy_v)
                select_weigth[0, 2, depth_index, lon_index, lat_index] = 1.0

            if 'CHLA' in var_list:
              chla_v = chla[channel_index]
              if not -5 < chla_v < 600:
                print('invalid chla found', chla_v)
              else:
                select_parallelepiped[0, 3, depth_index, lon_index, lat_index] = float(chla_v)
                select_weigth[0, 3, depth_index, lon_index, lat_index] = 1.0

  return

def insert_model_n1p_values(year, lat_limits, lon_limits, depth_limits, year_limits, resolution):
  """
    function that update the parallelepiped updating all the voxel with MODEL information
    year = folder of the year we are considering
    lat_limits = (lat_min, lat_max)
    lon_limits = (lon_min, lon_max)
    depth_limits = (depth_min, depth_max) in km
    year_limits = (year_min, year_max)
    resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
  """
  lat_min, lat_max = lat_limits
  lon_min, lon_max = lon_limits
  depth_min, depth_max = depth_limits
  year_min, year_max = year_limits
  w_res, h_res, d_res = resolution

  w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
  h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
  d = int((depth_max - depth_min) / d_res + 1)

  path_chl = os.getcwd() + "/dataset/MODEL/" + str(year) + '/N1p/'
  chl_files = os.listdir(path_chl)
  for model_file in chl_files:
    if model_file[0:3] != 'ave':
      continue
    file_chl = path_chl + model_file
    ds_chl = nc.Dataset(file_chl)

    time = model_file[4:12]
    time = read_date_time_sat(time)
    if not year_min < time < year_max:
      continue
    index = list_data_time.index(time) # index input tens considered, i.e. the one to upd
    select_parallelepiped = list_parallelepiped[index] # parall we are modifying

    latitude_list = ds_chl['lat'][:].data
    longitude_list = ds_chl['lon'][:].data
    depth_list = ds_chl['depth'][:].data

    chl_tens = torch.tensor(ds_chl['N1p'][:].data)[0, :, :, :] # tensor indexes as temp(depth, x, y)

    print(model_file + ' analysis started')
    for i in range(len(latitude_list)): # indexing over the latitude (3rd component of the tensor)
      print('n1p')
      for j in range(len(longitude_list)): # indexing over the longitude (2nd component of the tensor)
        for k in range(len(depth_list)): # indexing over the depth (1st component of the tensor)
          latitude = latitude_list[i]
          longitude = longitude_list[j]
          depth = depth_list[k]

          chl = float(chl_tens[k, i, j].item())

          if lat_max > latitude > lat_min:
            if lon_max > longitude > lon_min:
              if depth_max > depth > depth_min:
                latitude_index = find_index(latitude, lat_limits, w)
                longitude_index = find_index(longitude, lon_limits, h)
                depth_index = find_index(depth, depth_limits, d)

                if -5 < chl < 600:
                  select_parallelepiped[0, 5, depth_index, longitude_index, latitude_index] = chl
    print(model_file + ' analysis completed')
    np.save(os.getcwd() + "/saved_parallelepiped/n1p_parallelepiped2016.npy", list_parallelepiped)

  return

def insert_model_n3n_values(year, lat_limits, lon_limits, depth_limits, year_limits, resolution):
  """
    function that update the parallelepiped updating all the voxel with MODEL information
    year = folder of the year we are considering
    lat_limits = (lat_min, lat_max)
    lon_limits = (lon_min, lon_max)
    depth_limits = (depth_min, depth_max) in km
    year_limits = (year_min, year_max)
    resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
  """
  lat_min, lat_max = lat_limits
  lon_min, lon_max = lon_limits
  depth_min, depth_max = depth_limits
  year_min, year_max = year_limits
  w_res, h_res, d_res = resolution

  w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
  h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
  d = int((depth_max - depth_min) / d_res + 1)

  path_chl = os.getcwd() + "/dataset/MODEL/" + str(year) + '/N3n/'
  chl_files = os.listdir(path_chl)
  for model_file in chl_files:
    if model_file[0:3] != 'ave':
      continue
    file_chl = path_chl + model_file
    ds_chl = nc.Dataset(file_chl)

    time = model_file[4:12]
    time = read_date_time_sat(time)
    if not year_min < time < year_max:
      continue
    index = list_data_time.index(time) # index input tens considered, i.e. the one to upd
    select_parallelepiped = list_parallelepiped[index] # parall we are modifying

    latitude_list = ds_chl['lat'][:].data
    longitude_list = ds_chl['lon'][:].data
    depth_list = ds_chl['depth'][:].data

    chl_tens = torch.tensor(ds_chl['N3n'][:].data)[0, :, :, :] # tensor indexes as temp(depth, x, y)

    print(model_file + ' analysis started')
    for i in range(len(latitude_list)): # indexing over the latitude (3rd component of the tensor)
      print('N3n')
      for j in range(len(longitude_list)): # indexing over the longitude (2nd component of the tensor)
        for k in range(len(depth_list)): # indexing over the depth (1st component of the tensor)
          latitude = latitude_list[i]
          longitude = longitude_list[j]
          depth = depth_list[k]

          chl = float(chl_tens[k, i, j].item())

          if lat_max > latitude > lat_min:
            if lon_max > longitude > lon_min:
              if depth_max > depth > depth_min:
                latitude_index = find_index(latitude, lat_limits, w)
                longitude_index = find_index(longitude, lon_limits, h)
                depth_index = find_index(depth, depth_limits, d)

                if -5 < chl < 600:
                  select_parallelepiped[0, 6, depth_index, longitude_index, latitude_index] = chl
    print(model_file + ' analysis completed')
    np.save(os.getcwd() + "/saved_parallelepiped/n3n_parallelepiped2016.npy", list_parallelepiped)

  return

def insert_model_r6c_values(year, lat_limits, lon_limits, depth_limits, year_limits, resolution):
  """
    function that update the parallelepiped updating all the voxel with MODEL information
    year = folder of the year we are considering
    lat_limits = (lat_min, lat_max)
    lon_limits = (lon_min, lon_max)
    depth_limits = (depth_min, depth_max) in km
    year_limits = (year_min, year_max)
    resolution = (w_res, h_res, d_res) dimension of a voxel (in km)
  """
  lat_min, lat_max = lat_limits
  lon_min, lon_max = lon_limits
  depth_min, depth_max = depth_limits
  year_min, year_max = year_limits
  w_res, h_res, d_res = resolution

  w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
  h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
  d = int((depth_max - depth_min) / d_res + 1)

  path_chl = os.getcwd() + "/dataset/MODEL/" + str(year) + '/R6c/'
  chl_files = os.listdir(path_chl)
  for model_file in chl_files:
    if model_file[0:3] != 'ave':
      continue
    file_chl = path_chl + model_file
    ds_chl = nc.Dataset(file_chl)

    time = model_file[4:12]
    time = read_date_time_sat(time)
    if not year_min < time < year_max:
      continue
    index = list_data_time.index(time) # index input tens considered, i.e. the one to upd
    select_parallelepiped = list_parallelepiped[index] # parall we are modifying

    latitude_list = ds_chl['lat'][:].data
    longitude_list = ds_chl['lon'][:].data
    depth_list = ds_chl['depth'][:].data

    chl_tens = torch.tensor(ds_chl['R6c'][:].data)[0, :, :, :] # tensor indexes as temp(depth, x, y)

    print(model_file + ' analysis started')
    for i in range(len(latitude_list)): # indexing over the latitude (3rd component of the tensor)
      print('R6c')
      for j in range(len(longitude_list)): # indexing over the longitude (2nd component of the tensor)
        for k in range(len(depth_list)): # indexing over the depth (1st component of the tensor)
          latitude = latitude_list[i]
          longitude = longitude_list[j]
          depth = depth_list[k]

          chl = float(chl_tens[k, i, j].item())

          if lat_max > latitude > lat_min:
            if lon_max > longitude > lon_min:
              if depth_max > depth > depth_min:
                latitude_index = find_index(latitude, lat_limits, w)
                longitude_index = find_index(longitude, lon_limits, h)
                depth_index = find_index(depth, depth_limits, d)

                if -5 < chl < 600:
                  select_parallelepiped[0, 7, depth_index, longitude_index, latitude_index] = chl
    print(model_file + ' analysis completed')
    np.save(os.getcwd() + "/saved_parallelepiped/r6c_parallelepiped2016.npy", list_parallelepiped)

  return

list_data_time = create_list_date_time(year_interval)

list_parallelepiped = [
  create_box(batch, number_channel, latitude_interval, longitude_interval, depth_interval, resolution) for i in
  range(len(list_data_time))]

list_weight_float = [
  create_box(batch, number_channel, latitude_interval, longitude_interval, depth_interval, resolution) for i in
  range(len(list_data_time))]

list_weight_sat = [
  create_box(batch, number_channel, latitude_interval, longitude_interval, depth_interval, resolution) for i in
  range(len(list_data_time))]

t = 't'
w = 'w'

if kindof == 'model2016':
  insert_model_temp_values(year, latitude_interval, longitude_interval, depth_interval, year_interval, resolution)
  print('temp value inserted')
  np.save(os.getcwd() + "/saved_parallelepiped/temp_parallelepiped2.npy", list_parallelepiped)
  #list_parallelepiped = np.load(os.getcwd() + "/saved_parallelepiped/phys_parallelepiped.npy",allow_pickle = True)
  insert_model_sal_values(year, latitude_interval, longitude_interval, depth_interval, year_interval, resolution)
  print('sal value inserted')
  np.save(os.getcwd() + "/saved_parallelepiped/sal_parallelepiped2.npy", list_parallelepiped)
  #list_parallelepiped = np.load(os.getcwd() + "/saved_parallelepiped/phys_parallelepiped.npy",allow_pickle = True)
  insert_model_chl_values(year, latitude_interval, longitude_interval, depth_interval, year_interval, resolution)
  print('chl value inserted')
  np.save(os.getcwd() + "/saved_parallelepiped/chl_parallelepiped2016.npy", list_parallelepiped)
  #list_parallelepiped = np.load(os.getcwd() + "/saved_parallelepiped/chl_parallelepiped.npy",allow_pickle = True)
  insert_model_doxy_values(year, latitude_interval, longitude_interval, depth_interval, year_interval, resolution)
  print('doxy value inserted')
  np.save(os.getcwd() + "/saved_parallelepiped/doxy_parallelepiped2016.npy", list_parallelepiped)
  #list_parallelepiped = np.load(os.getcwd() + "/saved_parallelepiped/doxy_parallelepiped2.npy",allow_pickle = True)
  insert_model_ppn_values(year, latitude_interval, longitude_interval, depth_interval, year_interval, resolution)
  print('ppn value inserted')
  np.save(os.getcwd() + "/saved_parallelepiped/ppn_parallelepiped2016.npy", list_parallelepiped)
  #list_parallelepiped = np.load(os.getcwd() + "/saved_parallelepiped/ppn_parallelepiped.npy",allow_pickle = True)
  insert_model_n1p_values(year, latitude_interval, longitude_interval, depth_interval, year_interval, resolution)
  print('n1p value inserted')
  np.save(os.getcwd() + "/saved_parallelepiped/n1p_parallelepiped2016.npy", list_parallelepiped)
  #list_parallelepiped = np.load(os.getcwd() + "/saved_parallelepiped/n1p_parallelepiped.npy",allow_pickle = True)
  insert_model_n3n_values(year, latitude_interval, longitude_interval, depth_interval, year_interval, resolution)
  print('n3n value inserted')
  np.save(os.getcwd() + "/saved_parallelepiped/n3n_parallelepiped2016.npy", list_parallelepiped)
  #list_parallelepiped = np.load(os.getcwd() + "/saved_parallelepiped/n3n_parallelepiped.npy",allow_pickle = True)
  insert_model_r6c_values(year, latitude_interval, longitude_interval, depth_interval, year_interval, resolution)
  print('r6c value inserted')
  np.save(os.getcwd() + "/saved_parallelepiped/r6c_parallelepiped2016.npy", list_parallelepiped)
  #list_parallelepiped = np.load(os.getcwd() + "/saved_parallelepiped/r6c_parallelepiped.npy",allow_pickle = True)

  save_routine(kindof, list_parallelepiped, list_data_time, year_interval, t)
  plot_routine(kindof, list_parallelepiped, list_data_time, channels, year_interval, t)
