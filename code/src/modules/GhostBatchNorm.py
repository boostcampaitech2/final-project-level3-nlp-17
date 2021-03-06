import torch
import torch.nn as nn
import numpy as np

import unittest


class GhostBatchNorm(nn.Module):
    def __init__(self, input_dim, virtual_batch_size, momentum):
        super(GhostBatchNorm, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)


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
