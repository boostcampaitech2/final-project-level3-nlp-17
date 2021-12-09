import torch
import torch.nn as nn
import numpy as np

import unittest


class GhostBatchNorm(nn.Module):
    def __init__(self, num_features, num_splits, momentum=0.1):
        super(GhostBatchNorm, self).__init__()
        self.num_splits = num_splits
        self.num_features = num_features
        self.bn_list = [
            nn.BatchNorm1d(num_features=num_features, momentum=momentum)
            for _ in range(self.num_splits)
        ]

    def forward(self, input):
        batch_size = input.shape[0]

        assert ((batch_size % self.num_splits) == 0) or (
            self.num_features == input.shape[1]
        )
        part_len = batch_size // self.num_splits

        part = input[0:part_len, :]
        result = self.bn_list[0](part)
        for i in range(1, self.num_splits):
            part = input[i * part_len : (i + 1) * part_len, :]
            result = torch.cat([result, self.bn_list[i](part)], dim=0)

        return result


class GhostBatchNormTests(unittest.TestCase):
    def setUp(self):
        self.gbn = GhostBatchNorm(8, 3, 0.5)

    def test_runs(self):
        test_input = torch.tensor(
            [
                [1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
                [2.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [3.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [4.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [5.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [6.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [2.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [2.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
                [2.0, 0.0, 100.0, 200.0, 0.0, 1.0, 2.0, 3.0],
            ]
        )
        test_output = self.gbn(test_input)
        print(test_output)
        # self.assertEqual(list(test_output), [0., 0., 1.])


if __name__ == "__main__":
    unittest.main()
