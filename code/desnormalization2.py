import os
import torch
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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

def ARE(actual, predicted, land):
    numerator = torch.abs(actual - predicted)
    denominator = torch.abs(actual)

    W, H = denominator.shape
    error = torch.empty(W, H)

    # Avoid division by zero
    for w in range(W):
    	for h in range(H): 
    		if denominator[w, h] < land:
    			#print('denominator is 0')
    			#error[w, h] = numerator[w, h]
    			error[w, h] = 0
    		else:
    			error[w, h] = numerator[w, h] / denominator[w, h] * 100

    return error


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

	#target

	target = torch.load(path_directory + '/result_hmhsa/model2015/1test/tensor/target/' + dict_title[i] + '.pt', map_location=torch.device('cpu')) 
	#target = torch.load(path_configuration + '/tensor/target/' + dict_channel[i] + '.pt', map_location=torch.device('cpu')) 
	desnorm_target = (target * std_tensor[:,3+i,:,:,:]) + mean_tensor[:,3+i,:,:,:]
	number_fig = len(target[0, 0, :, 0, 0])  # number of levels of depth

	path_fig_to= path_configuration + '/desnorm_targetvsoutput/'
	if not os.path.exists(path_fig_to):
		os.mkdir(path_fig_to)

		#cmap max and min and measure symbol
	if i == 0:
		colorbar_lab = '$mg/m^3$'
		title = 'Chlorophyll-a ['
		title_are = 'ARE - Chlorophyll-a ['
		land = 1e-9
	elif i == 1:
		colorbar_lab = '$mg/m^3/day$'
		title = 'Net Primary Production ['
		title_are = 'ARE - Net Primary Production ['
		land = 1e-08
	elif i == 2:
		colorbar_lab = '$mmol/m^3$'
		title = 'Phosphate ['
		title_are = 'ARE - Phosphate ['
		land = 1e-8
	elif i == 3:
		colorbar_lab = '$mmol/m^3$'
		title = 'Nitrate ['
		title_are = 'ARE - Nitrate ['
		land = 1e-7
	elif i == 4:
		colorbar_lab = '$mg/m^3$'
		title = 'Medium Particulate Carbon ['
		title_are = 'ARE - Medium Particulate Carbon ['
		land = 1e-6

	for d in range(number_fig):
		path_fig_channel = path_fig_to + str(dict_channel[i])
		if not os.path.exists(path_fig_channel):
			os.mkdir(path_fig_channel)

		#plt.rcParams['font.family'] = "Arial"

		minv = np.min([torch.min(desnorm_target[0, 0, d, :, :]), torch.min(desnorm_output[0, 0, d, :, :])])
		maxv = np.max([torch.max(desnorm_target[0, 0, d, :, :]), torch.max(desnorm_output[0, 0, d, :, :])])
		#norm = TwoSlopeNorm(vmin=minv, vcenter=0, vmax=maxv)

		cmap = plt.get_cmap('Greens')
		fig, axs = plt.subplots(1, 2)
		im = axs[0].imshow(desnorm_target[0, 0, d, :, :], cmap=cmap, vmin=minv, vmax=maxv) #norm=norm)
		axs[0].set_title('Target')
		im = axs[1].imshow(desnorm_output[0, 0, d, :, :], cmap=cmap, vmin=minv, vmax=maxv) #norm=norm)
		axs[1].set_title('Output')		    
		for ax in axs.flat:
		    ax.set(xlabel='Latitude (Pixels)', ylabel='Longitude (Pixels)')
		for ax in axs.flat:
		    ax.label_outer()		    
		fig.suptitle(title + str(d*20) + 'm deep]', weight='bold')
		cbar_ax = fig.add_axes([0.95, 0.2, 0.02, 0.6])
		fig.colorbar(im, cax=cbar_ax, label=colorbar_lab)
		plt.savefig(path_fig_channel + "/profondity_level_" + str(d) + ".png", bbox_inches='tight')
		plt.close()


	#are

	path_fig_are= path_configuration + '/desnorm_ARE/'
	if not os.path.exists(path_fig_are):
		os.mkdir(path_fig_are)
	number_fig = len(desnorm_target[0, 0, :, 0, 0])

	for d in range(number_fig):
		path_fig_channel = path_fig_are + str(dict_channel[i])
		if not os.path.exists(path_fig_channel):
			os.mkdir(path_fig_channel)

		cmap = plt.get_cmap('Greens')
		plt.imshow(ARE(desnorm_target[0, 0, d, :, :], desnorm_output[0, 0, d, :, :], land), cmap=cmap, vmin=0, vmax=100)
		np.save(path_fig_channel + 'are_m',ARE(desnorm_target[0, 0, d, :, :], desnorm_output[0, 0, d, :, :],land))
		plt.title(title_are + str(d*20) + 'm deep]', weight='bold')
		plt.xlabel("Latitude (Pixels)")
		plt.ylabel("Longitude (Pixels)")
		cbar = plt.colorbar(label=colorbar_lab, ticks=[0, 20, 40, 60, 80, 100])
		cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
		plt.savefig(path_fig_channel + "/profondity_level_" + str(d) + ".png")
		plt.close()


	#difference

	diff = torch.abs(desnorm_target[0, 0, :, :, :] - desnorm_output[0, 0, :, :, :])
	path_fig_diff = path_configuration + '/difference/'
	if not os.path.exists(path_fig_diff):
		os.mkdir(path_fig_diff)
	np.save(path_fig_diff + dict_channel[i] + "dif.npy", diff)
	number_fig = len(diff[:, 0, 0])

	#cmap max and min and measure symbol
	if i == 0:
		#pf_min, pf_max = -1, 1
		colorbar_lab = '$mg/m^3$'
		title = 'Difference - Chlorophyll-a ['
	elif i == 1:
		#pf_min, pf_max = -1, 1
		colorbar_lab = '$mg/m^3/day$'
		title = 'Difference - Net Primary Production ['
	elif i == 2:
		#pf_min, pf_max = -1, 1
		colorbar_lab = '$mmol/m^3$'
		title = 'Difference - Phosphate ['
	elif i == 3:
		#pf_min, pf_max = -1, 1
		colorbar_lab = '$mmol/m^3$'
		title = 'Difference - Nitrate ['
	elif i == 4:
		#pf_min, pf_max = -1, 1
		colorbar_lab = '$mg/m^3$'
		title = 'Difference - Medium Particulate Carbon ['

	for d in range(number_fig):
		path_fig_channel = path_fig_diff + str(dict_channel[i])
		if not os.path.exists(path_fig_channel):
			os.mkdir(path_fig_channel)
		#plt.rcParams['font.family'] = "Arial"
		cmap = plt.get_cmap('BrBG')
		plt.imshow(diff[d,:,:], cmap=cmap)#, vmin=pf_min, vmax=pf_max)
		plt.title(title + str(d*20) + 'm deep]', weight='bold')
		plt.xlabel("Latitude (Pixels)")
		plt.ylabel("Longitude (Pixels)")
		plt.colorbar(label=colorbar_lab)
		plt.savefig(path_fig_channel + "/profondity_level_" + str(d) + ".png")
		plt.close()

#images

for i in range(5):
	dict_channel = {0: 'chla', 1: 'ppn', 2: 'nit', 3: 'amo', 4: 'r6c'}

	for l in range(3):
		dict_title = {0: 'temperature', 1: 'salinity', 2: 'oxygen'}
		path_configuration =  path_directory + '/result_hmhsa/model2015/2000ep_' + dict_channel[i] + 'tt'
		images = torch.load(path_directory + '/result_hmhsa/model2015/1test/tensor/images/images.pt', map_location=torch.device('cpu')) 
		#images = torch.load(path_configuration + '/tensor/images/images.pt', map_location=torch.device('cpu')) 
		desnorm_image = (images[:,l,:,:,:] * std_tensor[:,l,:,:,:]) + mean_tensor[:,l,:,:,:]
		number_fig = len(images[0, 0, :, 0, 0])  # number of levels of depth
		path_fig = path_configuration + '/desnorm_images/'
		if not os.path.exists(path_fig):
			os.mkdir(path_fig)

		#cmap max and min and measure symbol
		if l == 0:
			colorbar_lab = '$°C$'
			title = 'Input - Temperature ['
		elif l == 1:
			colorbar_lab = '$mg/m^3/day$'
			title = 'Input - Salinity ['
		elif l == 2:
			colorbar_lab = '$PSU$'
			title = 'Input - Oxygen ['

		for d in range(number_fig):
			path_fig_channel = path_fig + str(dict_title[l])
			if not os.path.exists(path_fig_channel):
				os.mkdir(path_fig_channel)
			#plt.rcParams['font.family'] = "Arial"
			cmap = plt.get_cmap('Greens')
			plt.imshow(desnorm_image[0, d, :, :], cmap=cmap)
			plt.title(title + str(d*20) + 'm deep]', weight='bold')
			plt.xlabel("Latitude (Pixels)")
			plt.ylabel("Longitude (Pixels)")
			plt.colorbar(label=colorbar_lab)
			plt.savefig(path_fig_channel + "/profondity_level_" + str(d) + ".png")
			plt.close()