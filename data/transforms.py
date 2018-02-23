import torch
import numpy as np

__all__ = ['WhiteningImage', 'WhiteningTriplet', 'TensorTriplet', 'TensorImage']


class Whitening(object):
    def whiten(self, x, side=0):
        x = np.float32(x)
        x /= self.scale
        x -= self.mean[side]
        x /= self.std[side]
        return x

    def __call__(self, data):
        raise NotImplementedError()


class WhiteningTriplet(Whitening):
    def __init__(self, *, mean=(0., 0.), std=(1., 1.), scale=255.):
        self.mean = np.array(mean[0])[:, None, None], np.array(mean[1])[:, None, None]
        self.std = np.array(std[0])[:, None, None], np.array(std[1])[:, None, None]
        self.scale = scale

    def __call__(self, data):
        R, P, Q = data
        return self.whiten(R, 0), self.whiten(P, 1), self.whiten(Q, 1)


class TensorTriplet(object):
    def __call__(self, data):
        R, P, Q = data
        R = torch.from_numpy(R)
        P = torch.from_numpy(P)
        Q = torch.from_numpy(Q)
        return R, P, Q


class WhiteningImage(Whitening):
    def __init__(self, *, mean=(0., 0.), std=(1., 1.), scale=255.):
        self.mean = np.array(mean[0])[None, None, :], np.array(mean[1])[None, None, :]
        self.std = np.array(std[0])[None, None, :], np.array(std[1])[None, None, :]
        self.scale = scale

    def __call__(self, data):
        L, R, D = data
        return self.whiten(L, 0), self.whiten(R, 1), D


class TensorImage(object):
    def __call__(self, data):
        L, R, D = data
        L = torch.from_numpy(L.transpose((2, 0, 1)))
        R = torch.from_numpy(R.transpose((2, 0, 1)))
        D = torch.from_numpy(D)
        return L, R, D
