import time
import random
import math
import logging
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import random_split
from tqdm.notebook import tnrange, tqdm

def save_params(params, filename):
    """This function intended to save params information generated during training.
    However, it is not successful at this moment when i am writing this docstring.
    I have not got time to solve how to save datetime/timedelta value into json.

    Args:
        params (dict): params information generated
        filename (string): path to json file where params should be saved

    Raises:
        ValueError: if cannot save
    """
    with open(filename, 'w') as file:
        try:
            json.dump(params, file)
        except:
            raise ValueError(f"Cannot save params into json. We need params for testing.")

#AE structure
class AutoEncoder(nn.Module):
    def __init__(self, first_layer, layer_1, layer_2):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(first_layer, layer_1),
            nn.ReLU(),
            
            nn.Linear(layer_1,layer_2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(layer_2, layer_1),
            nn.ReLU(),
            
            nn.Linear(layer_1, first_layer)            
        )

    def forward(self, x):
        
        x_encoder = self.encoder(x)
        x_decoder = self.decoder(x_encoder)

        return x_encoder,x_decoder
    
#custom dataset
class ChronosDataset(Dataset):
    def __init__(self,dat):
        self.dat = dat
        
    def __len__(self):
        return len(self.dat)
    
    def __getitem__(self,idx):
        return self.dat[idx]

#Trainning function
def _fit(ae, train_loader, val_loader, epochs, batch_size, tolerence, optimizer, lr, scheduler, loss_f, writer):
    """This is the training function of AE model

    Args:
        ae (nn.Module): autoencoder model
        train_loader (DataLoader): training data set
        val_loader (Dataloader): validation data set
        epochs (int): epochs number
        batch_size (int): batch size
        tolerence (int): qualification of early stopping, if val loss exceed best val loss for tolerence % then early stop.
        optimizer (torch.optim.Adam): adam optimizer
        lr (double): learning rate
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Adjust learning rate according to loss
        loss_f (nn.MSELoss()): Loss function, just mean square loss between real data and prediction
        writer (SummaryWriter): Tensorboard Writer
    """
    best_loss = 0.0
    for epoch in range(epochs):
        average_loss = 0.0
        for batchidx, x in enumerate(train_loader):
            x = Variable(x.float())
            _x_encoded, _x_decoded = ae(x.float())
            loss = loss_f(_x_decoded,x)
            # writer.add_scalar(f'loss/loss_{tolerence}_{epochs}_{lr}',loss, epoch * batch_size + batchidx)
            average_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_loss = average_loss / batch_size
        
        #Validation
        with torch.no_grad():
            average_val_loss = 0.0
            for batchidx, x in enumerate(val_loader):
                ae.eval()
                _x_encoded, _x_decoded = ae(x.float())
                val_loss = loss_f(_x_decoded,x)
                average_val_loss += val_loss.item()
                # writer.add_scalar(f'val_loss/val_loss_{tolerence}_{epochs}_{lr}',val_loss, epoch * batch_size + batchidx)
            average_val_loss = average_val_loss / batch_size
        
        if epoch % 1000 == 0:
            print(f'epoch number:{epoch} train_loss: {average_loss} val_loss: {average_val_loss}')
        scheduler.step(average_loss)
         
        # Early Stopping: Measure how much greater is current validation loss to best validation loss so far
        # (in percentage), if greater than tolerence than stop.
        # method comes from Early Stopping – But When? by Lutz Prechelt
        if epoch == 0:
            best_loss = average_val_loss 
        elif epoch > 0:
            generalization_loss = 100 * ((average_val_loss / best_loss) - 1)
            if generalization_loss > tolerence:
                logging.info(f"Early stopping is triggered because gl {generalization_loss} exceeds tolerence{tolerence}, current best loss: {best_loss}")
                break
            
        if average_val_loss < best_loss:
            best_loss = average_val_loss

    logging.info(f"Training finished, epoch runned: {epoch}, current best loss: {best_loss}")
    return ae, best_loss, epoch

def train_model(dat, batch_size, lr, epochs, tolerence, filename):
    """High level training process.

    Args:
        dat (numpy array): training input
        batch_size (int): pre-defined batch size
        lr (double): pre-defined learning rate
        epochs (int): pre-defined epochs number
        tolerence (int): pre-defined early stopping criteria
        filename (string): file path for saving AE models

    Raises:
        ValueError: cannot save models
    """
    from tqdm.notebook import trange, tqdm
    #trainning process

    #prepare data
    train_len = int(len(dat) * 0.8)
    val_len = len(dat) - train_len
    train_dat, val_dat = random_split(dat, [train_len, val_len])
    writer = SummaryWriter()
    train_loader = DataLoader(ChronosDataset(train_dat), batch_size, True)
    val_loader = DataLoader(ChronosDataset(val_dat), batch_size, True)
    first_layer = dat.shape[1]
    layer_1 = max(math.floor(first_layer/4),30) #max(math.floor(n_inputs/4), 30)
    layer_2 = max(math.floor(layer_1/4),15) #max(math.floor(n_units_1/4), 15)
    ae = AutoEncoder(first_layer, layer_1, layer_2)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5, patience=125, verbose=True)
    loss_f = nn.MSELoss()

    ae, _, _ = _fit(ae, train_loader, val_loader, epochs, batch_size, tolerence, optimizer, lr, scheduler, loss_f, writer)
    
    try:
        torch.save(ae,filename)
    except:
        raise ValueError(f'Could not save ae model into {filename}.')


class TrainPytorchAE():
    def __init__(self, batch_size, lr, epochs, tolerence, filename="saved_models/"):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.tolerence = tolerence
        self.filename =filename
        self.name = 'train_ae'
    
    def fit_transform(self, dfs, **params):
        """This function proceeds training step in pipeline.
        Weekdays and weekends data go through separate models.
        So two models will be trained and saved.

        Args:
            dfs (list of numpy arrays): training data.

        Raises:
            ValueError: if list of dataframes is empty.

        Returns:
            ae_list: file paths of saved models
        """
        if len(dfs) == 0:
            raise ValueError(f"There is no data inputs which required for step {self.name}")

        ae_list = []
        for i in range(len(dfs)):
            dat = dfs[i]
            if i:
                current_pattern = 'weekends'
            else:
                current_pattern = 'weekdays'
            filename = self.filename + f'pytorch_model_{current_pattern}'
            train_model(dat, self.batch_size, self.lr, self.epochs, self.tolerence, filename)
            ae_list.append(filename)
        # params_location = self.filename + f'params.json'
        # save_params(params, params_location)
        
        return ae_list, params