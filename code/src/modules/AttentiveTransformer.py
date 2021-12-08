import torch
import torch.nn as nn
import numpy as np

from src.modules.activations import Sparsemax
import unittest

class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, Na, gamma=1):
        super(AttentiveTransformer,self).__init__()
        self.gamma=gamma
        self.fc = nn.Linear(Na, input_dim)
        self.bn = nn.BatchNorm1d(input_dim) 
        self.sparsemax = Sparsemax()
    
    def forward(self, x, prior_scales):
        prior_bn = self.bn(self.fc(x)) * prior_scales
        mask = self.sparsemax(prior_bn)
        prior_scales = prior_scales * (self.gamma-mask)
        return mask, prior_scales

