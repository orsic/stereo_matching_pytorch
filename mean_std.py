import numpy as np
from pathlib import Path
from torchvision.transforms import Compose

from data.dataset import KittiDataset
from data.paths import kitti_paths
from data.transforms import WhiteningImage

if __name__ == '__main__':
    trans = Compose(
        [WhiteningImage()]
    )
    dataset = KittiDataset(kitti_paths((Path('/home/morsic/datasets/kitti_flow/training'))), transforms=trans)

    mean_l = np.zeros((3,))
    std_l = np.zeros((3,))

    mean_r = np.zeros((3,))
    std_r = np.zeros((3,))

    for triplet in dataset:
        L, R, _ = triplet
        mean_l += np.mean(L, axis=(0, 1))
        std_l += np.std(L, axis=(0, 1))

        mean_r += np.mean(R, axis=(0, 1))
        std_r += np.std(R, axis=(0, 1))

    print(f'Mean left: {mean_l / len(dataset)}')
    print(f'Std left: {std_l / len(dataset)}')
    print(f'Mean right: {mean_r / len(dataset)}')
    print(f'Std right: {std_r / len(dataset)}')
