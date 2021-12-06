import torch
import torch.nn as nn
import numpy as np

from src.modules.activations import Sparsemax
import unittest

class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, Na):
        super(AttentiveTransformer,self).__init__()
        self.fc = nn.Linear(Na, input_dim)
        self.bn = nn.BatchNorm1d(input_dim) 
        self.sparsemax = Sparsemax()
    
    def forward(self, x, prior_scales):
        prior_bn = self.bn(self.fc(x)) * prior_scales
        sparsemax = self.sparsemax(prior_bn)
        prior_scales = prior_scales * sparsemax
        return sparsemax, prior_scales

