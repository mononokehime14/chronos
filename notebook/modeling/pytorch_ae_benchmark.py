import time
import random
import pandas as pd
import numpy as np
import math
import logging
import os


import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import random_split
from tqdm.notebook import trange, tqdm


from .pytorch_ae import AutoEncoder, ChronosDataset

def _fit(ae, train_loader, val_loader, epochs, batch_size, optimizer, lr, scheduler, loss_f, writer):
    """This function is benchmark testing.
    Without early stopping and with given learning rate, batch_size and epochs number,
    it just trains the model and record training loss and validation loss for further comparision.

    Args:
        ae (nn.Module): autoencoder model
        train_loader (DataLoader): training data set
        val_loader (Dataloader): validation data set
        epochs (int): epochs number
        batch_size (int): batch size
        optimizer (torch.optim.Adam): adam optimizer
        lr (double): learning rate
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Adjust learning rate according to loss
        loss_f (nn.MSELoss()): Loss function, just mean square loss between real data and prediction
        writer (SummaryWriter): Tensorboard Writer

    Returns:
        loss_list and val_loss_list: records of training loss and validation loss in epochs scale.
    """
    val_loss_list = []
    loss_list = []
    
    for epoch in range(epochs):
        average_loss = 0.0
        for batchidx, x in enumerate(train_loader):
            x = Variable(x.float())
            _x_encoded, _x_decoded = ae(x.float())
            loss = loss_f(_x_decoded,x)
            average_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_loss = average_loss / batch_size
        loss_list.append(average_loss)
        writer.add_scalar(f'loss/{batch_size}_{epochs}_{lr}',average_loss, epoch)
        
        #Validation
        with torch.no_grad():
            average_val_loss = 0.0
            for batchidx, x in enumerate(val_loader):
                ae.eval()
                _x_encoded, _x_decoded = ae(x.float())
                val_loss = loss_f(_x_decoded,x)
                average_val_loss += val_loss.item()
            average_val_loss = average_val_loss / batch_size
            val_loss_list.append(average_val_loss)
            writer.add_scalar(f'val_loss/{batch_size}_{epochs}_{lr}',average_val_loss, epoch)
        
        if epoch % 100 == 0:
            print(f'epoch number:{epoch} train_loss: {average_loss} val_loss: {average_val_loss}')
        scheduler.step(average_loss)
         

    logging.info(f"Training finished, epoch runned: {epoch}")
    return loss_list, val_loss_list

def complement_memory_list(lst, max_size):
    """Epochs number can be different, 
    loss, val_loss of shorter epochs will be filled with NaN in order to align with longer ones

    Args:
        lst (list): loss list or val list
        max_size ([type]): max length (epochs)

    Returns:
        list: complemented loss list or val list
    """
    if len(lst) == max_size:
        return lst
    
    nan_len = max_size - len(lst)
    complement = [None] * nan_len
    lst = lst + complement
    return lst

def model_benchmark(dat, epochs_list, lr_list, batchsize_list):
    """Grid search, test all combinations of hyperparameters

    Args:
        dat (numpy array): training data
        epochs_list (list): pool of epochs candidates
        lr_list (list): pool of learning rates candidates
        batchsize_list (list): pool of batchsizes candidates
    """

    #prepare data
    train_len = int(len(dat) * 0.8)
    val_len = len(dat) - train_len
    train_dat, val_dat = random_split(dat, [train_len, val_len])
    writer = SummaryWriter()
    first_layer = dat.shape[1]
    layer_1 = max(math.floor(first_layer/4),30)
    layer_2 = max(math.floor(layer_1/4),15) 

    loss_df = pd.DataFrame()
    val_loss_df = pd.DataFrame()
    max_epoch = max(epochs_list)
    
    for batch_size in tqdm(batchsize_list):
        for lr in tqdm(lr_list):
            for epochs in tqdm(epochs_list):
                train_loader = DataLoader(ChronosDataset(train_dat), batch_size, True)
                val_loader = DataLoader(ChronosDataset(val_dat), batch_size, True)
                ae = AutoEncoder(first_layer, layer_1, layer_2)
                optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5, patience=125, verbose=True)
                loss_f = nn.MSELoss()

                loss_list, val_loss_list = _fit(ae, train_loader, val_loader, epochs, batch_size, optimizer, lr, scheduler, loss_f, writer)
                loss_list = complement_memory_list(loss_list,max_epoch)
                val_loss_list = complement_memory_list(val_loss_list,max_epoch)
                column_name = f'{batch_size}_{lr}_{epochs}'
                loss_df[column_name] = loss_list
                val_loss_df[column_name] = val_loss_list

    return loss_df, val_loss_df


class Benchmark():
    def __init__(self, epochs_list, lr_list, batchsize_list):
        self.epochs_list = epochs_list
        self.lr_list = lr_list
        self.batchsize_list = batchsize_list
        self.name = 'benchmark'

    def _validate_lists(self, dfs=None):
        if dfs is not None:
            if len(dfs) == 0:
                raise ValueError('Data inputs are empty! Error occur at Benchmark part.')
        else:
            if (len(self.epochs_list) == 0) | (len(self.lr_list) == 0) | (len(self.batchsize_list) == 0):
                raise ValueError('Benchmark test need lists of epochs, learning rates and batchsizes which are not present.')
        
    def fit_transform(self, dfs, **params):
        """This function preceeds benchmark testing in pipeline

        Args:
            dfs (list of numpy arrays): training datasets

        Returns:
            loss and validation loss memory: records of losses for different combinations
        """
        self._validate_lists(dfs)
        self._validate_lists()
        loss_memory = []
        val_loss_memory = []
        for i in range(len(dfs)):
            dat = dfs[i]

            if len(dat) == 0:
                continue
            loss_df, val_loss_df = model_benchmark(dat, self.epochs_list, self.lr_list, self.batchsize_list)
            loss_memory.append(loss_df)
            val_loss_memory.append(val_loss_df)
        return loss_memory, val_loss_memory

    