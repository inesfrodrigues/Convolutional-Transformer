import os
import torch
import numpy as np 
import matplotlib.pyplot as plt
from mean_pixel_value import MV_pixel, std_pixel


path_directory = os.getcwd()
directory_tensor = path_directory + '/tensor/(12, 12, 20)/'

model_tensor = []
directory_float = directory_tensor + 'model2015/'
list_ptFIles = os.listdir(directory_float)
for ptFiles in list_ptFIles:
    if ptFiles != 'datetime_2015.5.pt' and ptFiles != 'datetime_2015.49.pt' and ptFiles != 'datetime_2015.51.pt' and ptFiles != 'datetime_2015.52.pt':
        my_tensor = torch.load(directory_float + ptFiles)
        model_tensor.append(my_tensor[:, :, :-1, :, :])
        
list_tensor = model_tensor

number_channel = 8

#desnormalize

mean_value_pixel = MV_pixel(list_tensor)
mean_tensor = torch.tensor(mean_value_pixel.reshape(1, number_channel, 1, 1, 1))
std_value_pixel = std_pixel(list_tensor)
std_tensor = torch.tensor(std_value_pixel.reshape(1, number_channel, 1, 1, 1))


number_channel_output = 5 #5

for i in range(number_channel_output):
	dict_channel = {0: 'chla', 1: 'ppn', 2: 'nit', 3: 'amo', 4: 'r6c'}
	dict_title = {0: 'chla', 1: 'ppn', 2: 'nitrate', 3: 'n3n', 4: 'r6c'}
	path_configuration =  path_directory + '/result_hmhsa/model2015/2000ep_' + dict_channel[i] + 'tt' 

	#output

	output = torch.load(path_configuration + '/tensor/epoch_1999/tensor_phase1.pt', map_location=torch.device('cpu'))  #remove _phase1
	desnorm_output = (output * std_tensor[:,3+i,:,:,:]) + mean_tensor[:,3+i,:,:,:] 
	number_fig = len(output[0, 0, :, 0, 0])  # number of levels of depth
	path_fig_output = path_configuration + '/desnorm_output/'
	if not os.path.exists(path_fig_output):
		os.mkdir(path_fig_output)

	for d in range(number_fig):
		path_fig_channel = path_fig_output + str(dict_channel[i])
		if not os.path.exists(path_fig_channel):
			os.mkdir(path_fig_channel)
		cmap = plt.get_cmap('Greens')
		plt.imshow(desnorm_output[0, 0, d, :, :], cmap=cmap)
		plt.title(dict_title[i])
		plt.colorbar()
		plt.savefig(path_fig_channel + "/profondity_level_" + str(d) + ".png")
		plt.close()

	#target

	target = torch.load(path_directory + '/result_hmhsa/model2015/1test/tensor/target/' + dict_title[i] + '.pt', map_location=torch.device('cpu')) 
	#target = torch.load(path_configuration + '/tensor/target/' + dict_channel[i] + '.pt', map_location=torch.device('cpu')) 
	desnorm_target = (target * std_tensor[:,3+i,:,:,:]) + mean_tensor[:,3+i,:,:,:]
	number_fig = len(target[0, 0, :, 0, 0])  # number of levels of depth
	path_fig_target = path_configuration + '/desnorm_target/'
	if not os.path.exists(path_fig_target):
		os.mkdir(path_fig_target)

	for d in range(number_fig):
		path_fig_channel = path_fig_target + str(dict_channel[i])
		if not os.path.exists(path_fig_channel):
			os.mkdir(path_fig_channel)
		cmap = plt.get_cmap('Greens')
		plt.imshow(desnorm_target[0, 0, d, :, :], cmap=cmap)
		plt.title(dict_title[i])
		plt.colorbar()
		plt.savefig(path_fig_channel + "/profondity_level_" + str(d) + ".png")
		plt.close()

	#difference

	diff = target[0, 0, :, :, :] - output[0, 0, :, :, :]
	path_fig_diff = path_configuration + '/difference/'
	if not os.path.exists(path_fig_diff):
		os.mkdir(path_fig_diff)
	np.save(path_fig_diff + dict_channel[i] + "dif.npy", diff)
	number_fig = len(diff[:, 0, 0])

	#cmap max and min
	#if i == 0:
	#	pf_min, pf_max = -1, 1
	#elif i == 1:
	#	pf_min, pf_max = -1, 1
	#elif i == 2:
	#	pf_min, pf_max = -1, 1
	#elif i == 3:
	#	pf_min, pf_max = -1, 1
	#elif i == 4:
	#	pf_min, pf_max = -1, 1

	for d in range(number_fig):
		path_fig_channel = path_fig_diff + str(dict_channel[i])
		if not os.path.exists(path_fig_channel):
			os.mkdir(path_fig_channel)
		cmap = plt.get_cmap('BrBG')
		plt.imshow(diff[d, :, :], cmap=cmap)#, vmin=pf_min, vmax=pf_max)
		plt.title("difference - " + dict_title[i])
		plt.colorbar()
		plt.savefig(path_fig_channel + "/profondity_level_" + str(d) + ".png")
		plt.close()

#images

for i in range(3):
	dict_channel = {0: 'chla', 1: 'ppn', 2: 'nit', 3: 'amo', 4: 'r6c'}
	dict_title = {0: 'temperature', 1: 'salinity', 2: 'oxygen'}
	path_configuration =  path_directory + '/result_hmhsa/model2015/2000ep_' + dict_channel[i] + 'tt'
	images = torch.load(path_directory + '/result_hmhsa/model2015/1test/tensor/images/images.pt', map_location=torch.device('cpu')) 
	#images = torch.load(path_configuration + '/tensor/images/images.pt', map_location=torch.device('cpu')) 
	desnorm_image = (images[:,i,:,:,:] * std_tensor[:,i,:,:,:]) + mean_tensor[:,i,:,:,:]
	number_fig = len(images[0, 0, :, 0, 0])  # number of levels of depth
	path_fig = path_configuration + '/desnorm_images/'
	if not os.path.exists(path_fig):
		os.mkdir(path_fig)

	for d in range(number_fig):
		path_fig_channel = path_fig + str(dict_title[i])
		if not os.path.exists(path_fig_channel):
			os.mkdir(path_fig_channel)
		cmap = plt.get_cmap('Greens')
		plt.imshow(desnorm_image[0, d, :, :], cmap=cmap)
		plt.title(dict_title[i])
		plt.colorbar()
		plt.savefig(path_fig_channel + "/profondity_level_" + str(d) + ".png")
		plt.close()