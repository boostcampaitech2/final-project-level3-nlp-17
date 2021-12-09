import torch
import torch.nn as nn
import numpy as np

from src.modules.AttentiveTransformer import AttentiveTransformer
from src.modules.FeatureTransformer import FeatureTransformer, SharedAcrossDecisionStep
import unittest


class Aggregation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Aggregation, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class TabNetEncoderStep(nn.Module):
    def __init__(self, input_dim, Sharedacrossdecisionstep, Nd, Na, gamma):
        super(TabNetEncoderStep, self).__init__()
        self.input_dim = input_dim
        self.Nd = Nd
        self.Na = Na
        self.a_tran = AttentiveTransformer(self.input_dim, Na, gamma)
        self.f_tran = FeatureTransformer(
            self.input_dim, Sharedacrossdecisionstep, NdNa=Nd + Na
        )
        self.relu = nn.ReLU()
        self.agg = Aggregation(Nd, input_dim)

    def forward(self, a, x, prior_scales):
        mask, prior_scales = self.a_tran(a, prior_scales)
        masked_x = mask * x
        d, a = torch.split(self.f_tran(masked_x), [self.Nd, self.Na], dim=-1)
        output = self.relu(d)
        agg = self.agg(output)
        return a, output, mask, agg, prior_scales


class TabNetEncoder(nn.Module):
    def __init__(
        self, batch_size, input_dim, output_dim, step_num, Nd=64, Na=64, gamma=1
    ):
        super(TabNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.step_num = step_num
        self.Nd = Nd
        self.Na = Na
        self.bn = nn.BatchNorm1d(self.input_dim)
        self.Sharedacrossdecisionstep = SharedAcrossDecisionStep(
            input_dim, NdNa=Nd + Na
        )
        self.f_tran = FeatureTransformer(
            self.input_dim, self.Sharedacrossdecisionstep, NdNa=Nd + Na
        )
        self.TabNetEncoderStep_list = [
            TabNetEncoderStep(input_dim, self.Sharedacrossdecisionstep, Nd, Na, gamma)
            for _ in range(step_num)
        ]
        self.init_prior_scales = torch.ones((batch_size, input_dim))
        self.fc = nn.Linear(Nd, output_dim)

    def forward(self, f, S=None):
        bn_f = self.bn(f)
        d, a = torch.split(self.f_tran(bn_f), [self.Nd, self.Na], dim=-1)
        total_agg = torch.zeros_like(f)
        embeding = torch.zeros_like(d)
        total_mask = None
        if S:
            prior_scales = self.init_prior_scales[: f.shape[0], :] * S[: f.shape[0], :]
        else:
            prior_scales = self.init_prior_scales[: f.shape[0], :]
        for i in range(self.step_num):
            a, d, mask, agg, prior_scales = self.TabNetEncoderStep_list[i](
                a, bn_f, prior_scales
            )
            total_agg += mask * agg
            embeding = embeding + d
            if total_mask != None:
                total_mask = torch.cat(
                    [
                        total_mask.view(-1, total_mask.shape[1], total_mask.shape[2]),
                        mask.view(-1, mask.shape[0], mask.shape[1]),
                    ],
                    dim=0,
                )
            else:
                total_mask = mask.view(-1, mask.shape[0], mask.shape[1])
        output = self.fc(embeding)
        return embeding, output, total_agg, total_mask


class TabNetDecoderStep(nn.Module):
    def __init__(self, input_dim, output_dim, Sharedacrossdecisionstep):
        super(TabNetDecoderStep, self).__init__()
        self.f_tran = FeatureTransformer(input_dim, Sharedacrossdecisionstep, input_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(self.f_tran(x))


class TabNetDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, step_num):
        super(TabNetDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.step_num = step_num
        self.Sharedacrossdecisionstep = SharedAcrossDecisionStep(input_dim, input_dim)
        self.TabNetDecoderStep_list = [
            TabNetDecoderStep(input_dim, output_dim, self.Sharedacrossdecisionstep)
            for _ in range(self.step_num)
        ]

    def forward(self, x):
        reconstructed_features = torch.zeros((x.shape[0], self.output_dim))
        for i in range(self.step_num):
            reconstructed_features = (
                reconstructed_features + self.TabNetDecoderStep_list[i](x)
            )
        return reconstructed_features


class TabNet(nn.Module):
    def __init__(
        self, batch_size, input_dim, output_dim, step_num, Nd=64, Na=64, gamma=1
    ):
        super(TabNet, self).__init__()
        self.encoder = TabNetEncoder(
            batch_size, input_dim, output_dim, step_num, Nd, Na, gamma
        )
        self.decoder = TabNetDecoder(Nd, input_dim, step_num)

    def forward(self, x, S=None):
        if S:
            embeding, output, total_agg, total_mask = self.encoder(x)
            recon = self.decoder(embeding)
            return embeding, output, total_agg, recon, total_mask
        else:
            embeding, output, total_agg, total_mask = self.encoder(x)
            recon = self.decoder(embeding)
            return embeding, output, total_agg, recon, total_mask


class TabnetTests(unittest.TestCase):
    def setUp(self):
        self.tabnet = TabNet(
            batch_size=16, input_dim=10, output_dim=2, step_num=3, Nd=5, Na=5
        )

    def test_runs(self):
        test_input = torch.tensor(
            [
                [1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 2.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 4.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 5.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 6.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 2.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 2.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 2.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 2.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 4.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 5.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 2.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 4.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
            ]
        )
        test_output = self.tabnet(test_input)
        print(test_output)  # embeding, output, agg, recon


if __name__ == "__main__":
    unittest.main()
