import argparse
import h5py
from pathlib import Path
from tqdm import tqdm

from data.patches import generate_examples
from data.dataset import KittiDataset
from data.paths import kitti_paths

parser = argparse.ArgumentParser(description='Create HDF5 dataset storage')
parser.add_argument('root', type=str, help='Dataset root', default='/home/morsic/datasets/kitti_flow/training')
parser.add_argument('--dest', type=str, help='Path to HDF5', default='kitti.hdf5')

if __name__ == '__main__':
    args = parser.parse_args()
    root = Path(args.root)

    paths = kitti_paths(root)
    dataset = KittiDataset(paths)

    offset, created = 0, False
    f = h5py.File(args.dest, "w")
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        patches = generate_examples(example)
        N, P, C, H, W = patches.shape
        if not created:
            dset = f.create_dataset('triplets', data=patches, maxshape=(None, P, C, H, W))
        else:
            new_dset = f.get('triplets')
            new_dset.resize(len(new_dset) + N, axis=0)
            new_dset[offset:offset + N, :, :, :, :] = patches
        created = True
        offset += N
    dset.flush()
