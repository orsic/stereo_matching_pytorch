from torch.utils.data import Dataset

from .util import open_image, load_disparity


class KittiDataset(Dataset):
    def __init__(self, paths, transforms=None):
        self.paths = paths
        self.trans = transforms if transforms is not None else lambda x: x

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        left = open_image(path['left'])
        right = open_image(path['right'])
        disparity = load_disparity(path['disparity'])
        return self.trans((left, right, disparity))


class H5Dataset(Dataset):
    def __init__(self, file, transforms=None, in_memory=False):
        self.file = file
        self.dset = self.file.get('triplets')
        if in_memory:
            self.dset = self.dset.value
        self.trans = transforms if transforms is not None else lambda x: x

    def __len__(self):
        return self.dset.shape[0]

    def __getitem__(self, item):
        R, P, Q = [self.dset[item][i] for i in range(3)]
        return self.trans((R, P, Q))
