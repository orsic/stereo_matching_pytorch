import torch
import shutil


class Logger(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def precision_th(gt, disp, *, max_disp=192, max_error=3, return_diff=False):
    sparse_mask = gt > 0
    max_mask = gt < max_disp
    mask = sparse_mask & max_mask
    diff = torch.abs(gt - disp)
    miss = diff > max_error
    hit = diff <= max_error
    n_hits, n_misses = torch.sum(hit[mask]), torch.sum(miss[mask])
    if return_diff:
        assert disp.size(0) == 1
        R = diff.new(diff.size()[1:3]).int().zero_()
        G = diff.new(diff.size()[1:3]).int().zero_()
        B = diff.new(diff.size()[1:3]).int().zero_()
        R[mask & miss] = 255
        G[mask & hit] = 255
        return n_hits, n_misses, torch.stack((R, G, B), dim=-1)
    return n_hits, n_misses


def save_checkpoint(state, model_dir, is_best, *, filename='checkpoint.pth.tar'):
    filepath = model_dir / filename
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, model_dir / 'model_best.pth.tar')


def num_params(model):
    params = model.parameters()
    total = 0
    for w in params:
        current = 1
        for si in w.size():
            current *= si
        total += current
    return total


def split_dataset_paths(paths, *, train_ratio=0.8):
    split_idx = int(train_ratio * len(paths))
    return paths[:split_idx], paths[:split_idx]
