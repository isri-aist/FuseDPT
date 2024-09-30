import torch
from torch import nn

import torch.nn.functional as F


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        alpha = 1e-5
        diff_log = torch.log(alpha+target[valid_mask]) - torch.log(alpha+pred[valid_mask])
        #print("diff log:", diff_log)
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss

class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        print(valid_mask.shape)
        diff = torch.abs(target - pred)
        
        diff = diff[valid_mask]
        print(diff.shape)
        delta = self.threshold * diff.max().data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0*delta ** 2, 0.)
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        loss = diff.mean()
        return loss
