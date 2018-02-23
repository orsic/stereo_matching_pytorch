import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CostVolumeDot(nn.Module):
    def __init__(self, *, max_disp=192, stem_strides=1, **kwargs):
        super(CostVolumeDot, self).__init__()
        self.stem_stride = stem_strides
        self.max_disp = max_disp

    def dot(self, L, R, dim):
        return (L * R).sum(dim, keepdim=True)

    def forward(self, un_l, un_r, direction=1):
        return torch.stack(
            [self.dot(un_l, un_r, dim=1)] + [
                self.dot(un_l, torch.cat((un_r[:, :, :, -d * direction:], un_r[:, :, :, 0:-d * direction]), dim=3),
                         dim=1) for d in
                range(1, self.max_disp // self.stem_stride)], dim=2)


class McCNN(nn.Module):
    def __init__(self, *, features=64, ksize=3, padding=0, in_channels=3, **kwargs):
        super(McCNN, self).__init__()
        self.unaries = nn.Sequential(
            nn.Conv2d(in_channels, features, ksize, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, ksize, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, ksize, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, ksize, padding=padding),
        )

    def forward(self, image):
        return F.normalize(self.unaries.forward(image), dim=1)


class StereoMcCNN(nn.Module):
    def __init__(self, unaries, volume):
        super(StereoMcCNN, self).__init__()
        self.unaries = unaries
        self.volume = volume
        self.criterion = None

    def forward(self, L, R):
        LU, RU = self.unaries.forward(L), self.unaries.forward(R)
        CV = self.volume.forward(LU, RU)
        max, argmax = torch.max(CV, dim=2, keepdim=True)
        return argmax.float()

    def loss(self, data):
        R, P, Q = data
        R, P, Q = [self.unaries.forward(Variable(patch).cuda(async=True)) for patch in (R, P, Q)]
        N = R.size(0)
        return self.criterion(R.view(N, -1), P.view(N, -1), Q.view(N, -1))
