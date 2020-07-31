import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.core.lightning import LightningModule

from moduels.model import *

class SAGAN(LightningModule):
    def __init__(self):
        super(LitModel, self).__init__()
        self.model = nn.Sequential()

    def forward(self, x):

    def training_step(self, batch, batch_idx):

        return{'loss': loss, 'log': }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=)