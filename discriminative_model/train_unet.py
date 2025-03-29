import os

import torch

from model.unet import FWIUNet, FWIUNetAtt
from model.segnext import FWINetwork
from train import train_unet_fwi, train_segnext_fwi
from utils.data_loader import dataloader_missing, dataloader_fwi, dataloader_fwi_extra

import warnings
warnings.filterwarnings("ignore") 

train_loader, test_loader = dataloader_fwi(batch_size=64)
fwinet = FWIUNet()
# fwinet.load_state_dict(torch.load('results/dcn_unet_fwi/model.pt'))

train_unet_fwi(fwinet, train_loader, test_loader, lr=5e-4, num_epochs=1000, patience=100)