import pdb

import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = OmegaConf.load('src/models/configs/model_conf.yaml')
        self.fc1 = nn.Linear(cfg.model.input_size, cfg.model.layer2_size)
        self.fc2 = nn.Linear(cfg.model.layer2_size, cfg.model.layer3_size)
        self.fc3 = nn.Linear(cfg.model.layer3_size, cfg.model.layer4_size)
        self.fc4 = nn.Linear(cfg.model.layer4_size, cfg.model.num_classes)
        self.dropout = nn.Dropout(p=cfg.model.dropout)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.dropout(self.fc4(x)), dim=1)
        
        return x
