import math
import sys
from typing import Iterable, Optional
import os 
from IPython import display
from plot_error import Plot_Error, Plot_Error_Both
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils
import numpy as np
import matplotlib.pyplot as plt

path='result_hmhsa/model2015' # result directory

epoch1=1

path_configuration = path + '/' + str(epoch1) + 'test' 
if not os.path.exists(path_configuration):
    os.mkdir(path_configuration)

torch.cuda.empty_cache()

losses_avg=[]
losses_test_avg=[]

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False):
    network_loss = torch.nn.MSELoss()
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    losses=[]


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=not fp32):
            outputs = model(samples)
            loss = network_loss(outputs, targets)
            losses.append(loss.item())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if epoch == epoch1-1:
            print('Final Loss of HMHSA Network: ', losses[-1])

    losses_avg.append((sum(losses)/len(losses)))


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch):
    network_loss = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    losses_test=[]

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = network_loss(output, target)
            losses_test.append(loss.item())

        
        path_tensor = path_configuration + '/tensor/'
        if not os.path.exists(path_tensor):
            os.mkdir(path_tensor)

        path_fig = path_configuration + '/fig/'
        if not os.path.exists(path_fig):
            os.mkdir(path_fig)

        path_tensor_output = path_tensor + 'output/'
        if not os.path.exists(path_tensor_output):
            os.mkdir(path_tensor_output)

        path_tensor_epoch = path_tensor_output + 'epoch_' + str(epoch)
        if not os.path.exists(path_tensor_epoch):
            os.mkdir(path_tensor_epoch)
        torch.save(output, path_tensor_epoch + "/output" + ".pt")

        path_fig_epoch = path_fig + 'epoch_' + str(epoch)
        if not os.path.exists(path_fig_epoch):
            os.mkdir(path_fig_epoch) 

        path_fig_original = path_fig + 'original_fig'
        if not os.path.exists(path_fig_original):
            os.mkdir(path_fig_original)


        number_fig = len(output[0, 0, :, 0, 0])  # number of levels of depth
        dict_channel = {0: 'temperature', 1: 'salinity', 2: 'oxygen', 3: 'chla', 4: 'ppn', 5: 'nitrate', 6: 'n3n', 7: 'r6c'}
        chan = 3

        for i in range(number_fig):
            #plot of output
            path_fig_channel = path_fig_epoch + '/' + str(dict_channel[chan])
            if not os.path.exists(path_fig_channel):
                os.mkdir(path_fig_channel)
            cmap = plt.get_cmap('Greens')
            output2 = output.cpu().detach().numpy()
            plt.imshow(output2[0, 0, i, :, :], cmap=cmap) 
            plt.title(dict_channel[chan])
            plt.colorbar()
            plt.savefig(path_fig_channel + "/profondity_level_" + str(i) + ".png")
            plt.close()

            #plots of other variables 
            for channel in [0, 1, 2]:
                if epoch == 0:
                    path_target = path_tensor + 'target/'
                    if not os.path.exists(path_target):
                        os.mkdir(path_target)
                    torch.save(target, path_target + "/" + str(dict_channel[chan]) + ".pt")

                    path_images = path_tensor + 'images/'
                    if not os.path.exists(path_images):
                        os.mkdir(path_images)
                    torch.save(images, path_images + "/images" + ".pt")

                    path_fig_channel = path_fig_original + '/' + str(dict_channel[channel])
                    if not os.path.exists(path_fig_channel):
                        os.mkdir(path_fig_channel)
                    images2 = images.cpu().detach().numpy()
                    plt.imshow(images2[0, channel, i, :, :], cmap=cmap) 
                    plt.title(dict_channel[channel])
                    plt.colorbar()
                    plt.savefig(path_fig_channel + "/profondity_level_original_" + str(i) + ".png")
                    plt.close()
                    path_fig_channel = path_fig_original + '/' + str(dict_channel[chan])
                    if not os.path.exists(path_fig_channel):
                        os.mkdir(path_fig_channel)
                    target2 = target.cpu().detach().numpy()
                    plt.imshow(target2[0, 0, i, :, :], cmap=cmap)
                    plt.title(dict_channel[chan])
                    plt.colorbar()
                    plt.savefig(path_fig_channel + "/profondity_level_original_" + str(i) + ".png")
                    plt.close()

        path_model = 'model_hmhsa/model2015/' + 'epoch_' + str(epoch1) + '.pt'
        torch.save(model.state_dict(), path_model)
        torch.save(model.state_dict(), path_configuration + '/model_hmhsa_' + 'epoch_' + str(epoch1) + '.pt')
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

    Plot_Error(losses_avg, 'train', path_configuration + '/')

    losses_test_avg.append((sum(losses_test)/len(losses_test)))
    Plot_Error(losses_test_avg, 'test', path_configuration + '/')

    Plot_Error_Both(losses_avg, losses_test_avg, path_configuration + '/')

    np.save(path_configuration + "/losses_test_avg.npy", losses_test_avg)
    np.save(path_configuration + "/losses_avg.npy", losses_avg)

    metric_logger.synchronize_between_processes()

    if epoch == epoch1-1:
        # printing final loss of testing set
        print('Final Loss TEST: ', losses_test[-1])

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
