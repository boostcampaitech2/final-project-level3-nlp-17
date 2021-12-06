import torch
import torch.nn as nn
import numpy as np

from src.modules.AttentiveTransformer import AttentiveTransformer
from src.modules.FeatureTransformer import FeatureTransformer, SharedAcrossDecisionStep
import unittest

class TabNetEncoderStep(nn.Module):
    def __init__(self, input_dim, Sharedacrossdecisionstep, Nd, Na):
        super(TabNetEncoderStep,self).__init__()
        self.input_dim = input_dim
        self.Nd = Nd
        self.Na = Na
        self.a_tran = AttentiveTransformer(self.input_dim, Na)
        self.f_tran = FeatureTransformer(self.input_dim, Sharedacrossdecisionstep, NdNa = Nd+Na)
        self.relu = nn.ReLU()

    def forward(self, a, x, prior_scales):
        mask, prior_scales = self.a_tran(a, prior_scales.clone())
        masked_x = mask*x
        d, a = torch.split(self.f_tran(masked_x), [self.Nd, self.Na], dim=-1)
        output = self.relu(d)
        return a, output, masked_x, prior_scales

    
class TabNetEncoder(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, S=None, step_num=5, Nd=64, Na=64):
        super(TabNetEncoder,self).__init__()
        self.input_dim = input_dim
        self.step_num = step_num
        self.Nd = Nd
        self.Na = Na
        self.bn = nn.BatchNorm1d(self.input_dim)
        self.Sharedacrossdecisionstep = SharedAcrossDecisionStep(input_dim, NdNa=Nd+Na)
        self.f_tran = FeatureTransformer(self.input_dim, self.Sharedacrossdecisionstep, NdNa = Nd+Na)
        self.TabNetEncoderStep_list = [TabNetEncoderStep(input_dim, self.Sharedacrossdecisionstep, Nd, Na) for _ in range(step_num)]
        if S:
            self.init_prior_scales = torch.ones((batch_size, input_dim)) * S
        else:
            self.init_prior_scales = torch.ones((batch_size, input_dim))
        self.fc = nn.Linear(Nd, output_dim)
    
    def forward(self, f):
        bn_f = self.bn(f)
        d, a = torch.split(self.f_tran(bn_f), [self.Nd, self.Na],dim=-1)
        agg = torch.zeros_like(a)
        embeding = torch.zeros_like(d)
        prior_scales = self.init_prior_scales[:f.shape[0],:]
        for i in range(self.step_num):
            a2, d, masked_x, prior_scales = self.TabNetEncoderStep_list[i](a, bn_f, prior_scales)
            agg = agg.clone()+a2*d
            embeding = embeding.clone() + d
        output = self.fc(embeding)
        return embeding, output, agg

class TabNetDecoderStep(nn.Module):
    def __init__(self, input_dim, Sharedacrossdecisionstep):
        super(TabNetDecoderStep,self).__init__()
        self.f_tran = FeatureTransformer(input_dim, Sharedacrossdecisionstep, input_dim)
        self.fc = nn.Linear(input_dim, input_dim)
    def forward(self, x):
        return self.fc(self.f_tran(x))

class TabNetDecoder(nn.Module):
    def __init__(self, input_dim, step_num):
        super(TabNetDecoder,self).__init__()
        self.input_dim = input_dim
        self.step_num = step_num
        self.Sharedacrossdecisionstep = SharedAcrossDecisionStep(input_dim, NdNa=input_dim)
        self.TabNetDecoderStep_list = [TabNetDecoderStep(input_dim, self.Sharedacrossdecisionstep) for _ in range(self.step_num)]
    def forward(self, x):
        reconstructed_features = torch.zeros((x.shape[0], self.input_dim))
        for i in range(self.step_num):
            reconstructed_features = reconstructed_features.clone() + self.TabNetDecoderStep_list[i](x)
        return reconstructed_features


class TabNet(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, S=None, step_num=5, Nd=64, Na=64):
        super(TabNet,self).__init__()
        self.encoder = TabNetEncoder(batch_size, input_dim, output_dim, S, 5, Nd, Na)
        self.decoder = TabNetDecoder(Nd, step_num)
    def forward(self, x):
        embeding, output, agg = self.encoder(x)
        recon = self.decoder(embeding)
        return embeding, output, agg, recon

class TabnetTests(unittest.TestCase):
    def setUp(self):
        self.tabnet = TabNet(batch_size=16, input_dim = 10, output_dim=2, step_num=3, Nd=5, Na=5)
    def test_runs(self):
        test_input = torch.tensor([
            [1., 2., 1., 1.,  2., 3., 1.,  2., 3., 4.],
            [1., 2., 2., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 3., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 4., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 5., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 6., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 2., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 2., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 2., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 2., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 3., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 4., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 5., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 2., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 3., 0.,  100., 200., 0.,  1., 2., 3.],
            [1., 2., 4., 0.,  100., 200., 0.,  1., 2., 3.]
            ])
        test_output = self.tabnet(test_input)
        print(test_output) #embeding, output, agg, recon

if __name__=='__main__':
    unittest.main()