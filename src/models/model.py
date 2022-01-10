import pdb

import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.input_size, cfg.layer2_size)
        self.fc2 = nn.Linear(cfg.layer2_size, cfg.layer3_size)
        self.fc3 = nn.Linear(cfg.layer3_size, cfg.layer4_size)
        self.fc4 = nn.Linear(cfg.layer4_size, cfg.num_classes)
        self.dropout = nn.Dropout(p=cfg.dropout)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.dropout(self.fc4(x)), dim=1)
        
        return x
