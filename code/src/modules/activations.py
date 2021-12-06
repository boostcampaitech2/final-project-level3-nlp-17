import torch
import torch.nn as nn
import numpy as np

import unittest

def sparsemax(z: torch.tensor, a: int = 1):
    z_sorted, _ = torch.sort(z, dim=0, descending=True)
    cssz = torch.cumsum(z_sorted, dim=0)
    ind = torch.arange(1, 1 + len(z)).to(dtype=z.dtype)
    cond = a + ind*z_sorted - cssz > 0
    rho = ind.masked_select(cond)[-1]
    tau = (cssz.masked_select(cond)[-1]-1) / rho
    w = torch.clamp(z - tau, min=0)
    return w

class Sparsemax(nn.Module):
    def __init__(self, a: int = 1):
        super().__init__()
        self.a = a
    def forward(self, z: torch.tensor):
        if len(z.shape)==2:
            result = sparsemax(z[0], self.a).view(1,-1)
            for x in z[1:]:
                result = torch.cat([result, sparsemax(x, self.a).view(1,-1)])
            return result 
        else:
            return sparsemax(z, self.a)


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.sparsemax = Sparsemax()
    
    def test_runs(self):
        test_input = torch.tensor([1.,  2., 3.])
        test_output = self.sparsemax(test_input)
        self.assertEqual(list(test_output), [0., 0., 1.])

if __name__=='__main__':
    unittest.main()
