import torch
import torch.nn as nn
import numpy as np

import unittest




class SharedAcrossDecisionStep(nn.Module):
    def __init__(self, input_dim, NdNa):
        super(SharedAcrossDecisionStep,self).__init__()

        self.fc1 = nn.Linear(input_dim, NdNa*2)
        self.bn1 = nn.BatchNorm1d(NdNa*2)
        self.glu1 = nn.GLU()

        
        self.fc2 = nn.Linear(NdNa, NdNa*2)
        self.bn2 = nn.BatchNorm1d(NdNa*2)
        self.glu2 = nn.GLU()
        
    def forward(self, x):

        #Shared across decision step
        glu1 = self.glu1(self.bn1(self.fc1(x)))
        glu2 = self.glu2(self.bn2(self.fc2(glu1)))

        sads = (glu1+glu2)*np.sqrt(0.5)
        return sads

class DecisionStepDependent(nn.Module):
    def __init__(self, NdNa):
        super(DecisionStepDependent,self).__init__()

        self.fc3 = nn.Linear(NdNa, NdNa*2)
        self.bn3 = nn.BatchNorm1d(NdNa*2)
        self.glu3 = nn.GLU()
        
        self.fc4 = nn.Linear(NdNa, NdNa*2)
        self.bn4 = nn.BatchNorm1d(NdNa*2)
        self.glu4 = nn.GLU()
        
    def forward(self, x):

        #Shared across decision step
        fc3 = self.fc3(x)
        bn3 = self.bn3(fc3)
        glu3 = self.glu3(bn3)
        dsd1 = (x+glu3)*np.sqrt(0.5)

        glu4 = self.glu4(self.bn4(self.fc4(dsd1)))
        dsd2 = (dsd1+glu4)*np.sqrt(0.5)
        return dsd2


     
class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, Sharedacrossdecisionstep, NdNa):
        
        super(FeatureTransformer,self).__init__()

        #Shared across decision step
        self.Sharedacrossdecisionstep = Sharedacrossdecisionstep

        #Decision step dependent
        self.Decisionstepdependent = DecisionStepDependent(NdNa)

    def forward(self, x):
        #Shared across decision step
        sads = self.Sharedacrossdecisionstep(x)

        #Decision step dependent
        dsd = self.Decisionstepdependent(sads)

        return dsd
