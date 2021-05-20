import time
import random
import math
import logging

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import random_split

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
            current_loss = 0.0
            for batchidx, x in enumerate(val_loader):
                ae.eval()
                _x_encoded, _x_decoded = ae(x.float())
                val_loss = loss_f(_x_decoded,x)
                current_loss += val_loss.item()
                # writer.add_scalar(f'val_loss/val_loss_{tolerence}_{epochs}_{lr}',val_loss, epoch * batch_size + batchidx)
            current_loss = current_loss / batch_size
#             if epoch == 0:
#                 best_loss = current_loss        
#             elif current_loss < best_loss:
#                 patience_level = 0
#                 best_model = ae
#                 best_loss = current_loss
#             else:
#                 patience_level += 1
        
        if epoch % 1000 == 0:
            print(f'epoch number:{epoch} train_loss: {average_loss} val_loss: {current_loss}')
        scheduler.step(average_loss)
         
        # Early Stopping: Measure how much greater is current validation loss to best validation loss so far
        # (in percentage), if greater than tolerence than stop.
        # method comes from Early Stopping – But When? by Lutz Prechelt
        if epoch == 0:
            best_loss = current_loss 
        elif epoch > 0:
            generalization_loss = 100 * ((current_loss / best_loss) - 1)
            if generalization_loss > tolerence:
                logging.info(f"Early stopping is triggered because gl {generalization_loss} exceeds tolerence{tolerence}, current best loss: {best_loss}")
                break
            
        if current_loss < best_loss:
            best_loss = current_loss
        
        #commented because following code is used for another method of early stopping
#         if patience_level >= tolerence:
#             logging.info(f"Early stopping is triggered because {patience_level}, current best loss: {best_loss}")
#             break

    logging.info(f"Training finished, epoch runned: {epoch}, current best loss: {best_loss}")
    return ae, best_loss, epoch

def train_model(dat, batch_size, lr, epochs, tolerence, filename):
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

    ae, best_loss, epoch = _fit(ae, train_loader, val_loader, epochs, batch_size, tolerence, optimizer, lr, scheduler, loss_f, writer)
    
    try:
        torch.save(ae,filename)
    except:
        raise ValueError(f'Could not save ae model into {filename}.')

    return ae

class TrainPytorchAE():
    def __init__(self, filename="saved_models/", batch_size=32, lr=0.001, epochs=1000, tolerence=4):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.tolerence = tolerence
        self.filename =filename
        self.name = 'train_ae'
    
    def fit_transform(self, dfs, **params):
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
            ae = train_model(dat, self.batch_size, self.lr, self.epochs, self.tolerence, filename)
            ae_list.append(ae)
        
        return ae_list