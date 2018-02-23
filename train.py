import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

import sys
import os
from pathlib import Path
import argparse
import h5py
from mpi4py import MPI

from training.util import Logger, num_params
from training.trainer import Trainer

from models import mccnn
from data import dataset
from data.paths import kitti_paths
from data.transforms import *

parser = argparse.ArgumentParser(description='Detector train')
parser.add_argument('root', type=str, help='Dataset root', default='/home/morsic/datasets/kitti_flow/training')
parser.add_argument('experiment_dir', type=str, help='Direcotry of experiment')
parser.add_argument('--train_data', type=str, help='Path to HDF5', default='/mnt/sdc/morsic/h5kitti/kitti.hdf5')

if __name__ == '__main__':
    args = parser.parse_args()

    root = Path(args.root)
    directory = Path(args.experiment_dir)
    directory.mkdir(exist_ok=True)
    model_dir = directory / 'checkpoints'
    model_dir.mkdir(exist_ok=True)

    config = {
        'max_disp': 192,
        'batch_size': 128,
        'lr': 1e-3,
        'features': 64,
        'ksize': 3,
        'padding': 0,
        'stem_strides': 1,
        'margin': 0.2,
        'epochs': 14,
    }

    unaries = mccnn.McCNN(**config)
    volume = mccnn.CostVolumeDot(**config)
    model = mccnn.StereoMcCNN(unaries, volume)
    mean = ([0.37944637, 0.39879613, 0.38400045], [0.38374098, 0.40427599, 0.3900857])
    std = ([0.29843408, 0.30832244, 0.31543067], [0.29652266, 0.30781594, 0.31405455])

    print(model)
    print('Number of parameters: {}'.format(num_params(model)))

    criterion = nn.TripletMarginLoss(config.get('margin', 0.1))
    model.criterion = criterion

    optimizer = Adam(model.parameters(), lr=config['lr'])

    kitti_paths = kitti_paths(root)
    dataset_splits = {}

    trans_train = Compose((WhiteningTriplet(mean=mean, std=std), TensorTriplet()))

    trans_val = Compose((WhiteningImage(mean=mean, std=std), TensorImage()))
    dataset_val = dataset.KittiDataset(kitti_paths, transforms=trans_val)

    with h5py.File(args.train_data, "r", driver='mpio', comm=MPI.COMM_WORLD) as h5_file:
        dataset_train = dataset.H5Dataset(h5_file, transforms=trans_train, in_memory=True)

    dataloader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    f = open(os.path.join(directory, 'train.txt'), 'w')
    sys.stdout = Logger(sys.stdout, f)

    model.cuda()
    model.train()

    with Trainer(model, dataloader_train, dataloader_val, optimizer, model_dir, **config) as trainer:
        trainer.train()
