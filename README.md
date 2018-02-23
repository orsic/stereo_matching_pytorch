# Stereo Matching Network
Original article: https://arxiv.org/abs/1510.05970

## Setup

Requirements:
* PyTorch
* H5Py ([with parallelism](http://docs.h5py.org/en/latest/mpi.html#parallel-hdf5)):

## Run

Create HDF5 dataset:
```bash
  python create_hdf5.py path/to/kitti/training --dest=path/to/kitti.h5
```

Train the model:
```bash
  python train.py path/to/kitti/training store/dir --train_data=path/to/kitti.h5
```

## TODO

HDF5 reading is slow. Data reader(and filesystem organisation) should be implemented to enable efficient data preprocessing.
Currently, whole dataset is loaded in memory to enable efficient GPu utilization(dataset is ~13GB).