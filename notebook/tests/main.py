import time
import random
import math
import logging
import numpy as np
import pandas as pd
import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import random_split


sys.path.append("..")
from preprocessing.preprocessor import LoadData, FindFrequency, PeriodDetect, AlignData, DropExtrema, Normalizer, FillGap, GenerateInput
from pipeline import Pipeline

#fname = "data/gt-weekly.zip"

#dscol = 'time'

# ycols = ['i1','i2','i3','i4','i5','i6','i7','i8','i9','i10']
print(os.getcwd())
data_path = 'for_tests/test_no_weekends.csv'
dscol = 'Time'
ycols = ["SA1900282"]
data_load = LoadData(data_path=data_path, dscol=dscol, ycols=ycols)
params = {}
dfs, _ = data_load.fit_transform(**params)

print(dfs[1].empty)

# test_task_list = [
#     LoadData(fname, dscol, ycols),
#     # FindFrequency(),
#     # PeriodDetect(),
#     # AlignData(),
#     # DropExtrema(),
#     # Normalizer(),
#     # FillGap(),
#     # GenerateInput()
# ]
# test_pipeline = Pipeline()
# dats, params = test_pipeline.transform(test_task_list)
# dat_weekdays = dats[0]
# dat_weekends = dats[1]