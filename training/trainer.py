import torch
from torch.autograd import Variable

import sys

from training.util import precision_th, save_checkpoint


class Trainer(object):
    def __init__(self, model, dataloader_train, dataloader_val, optimizer, model_dir, *,
                 epochs=1, batch_size=1, patch_size=9, lr=1e-3, lr_min=.0, checkpoint=None, max_disp=192,
                 max_error=3, lr_policy={11: 1e-4}, print_each=100, **kwargs):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.model_dir = model_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_steps = len(self.dataloader_train) * self.epochs // self.batch_size
        self.lr = lr
        self.lr_min = lr_min
        self.checkpoint = checkpoint
        self.max_disp = max_disp
        self.max_error = max_error
        self.lr_policy = lr_policy
        self.print_each = print_each
        self.crop = patch_size // 2
        self.best_validation = 0.0
        self.epoch_start = 0
        self.global_step = 0

    def __enter__(self):
        if self.checkpoint is not None:
            path = self.model_dir / self.checkpoint
            checkpoint = torch.load(path)
            print("Loading state from {}".format(path), file=sys.stderr)
            self.epoch_start = checkpoint['epoch']
            self.global_step = len(self.dataloader_train) * self.epoch_start // self.batch_size
            self.best_validation = checkpoint['best_validation']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save(-1, False)

    def set_learning_rate(self, epoch, lr_policy):
        if epoch in lr_policy:
            lr = lr_policy[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train(self):
        for epoch in range(self.epoch_start, self.epochs):
            try:
                # train steps for epoch
                self.model.train()
                for param_group in self.optimizer.param_groups:
                    print('LR: ', param_group['lr'], file=sys.stderr)
                for i, example in enumerate(self.dataloader_train):
                    self.optimizer.zero_grad()
                    self.set_learning_rate(epoch, self.lr_policy)
                    self.global_step += 1
                    loss = self.model.loss(example)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if i % self.print_each == 0:
                        print("train: epoch {} loss {}".format(epoch, loss.data[0]))
                # validation
                self.model.eval()
                is_best = False
                hits, misses = self.evaluate(self.dataloader_val, epoch)
                precision_val = 100 * hits / (hits + misses)
                print("{}: epoch {} total precision {}".format('train_valid', epoch, precision_val))
                if precision_val > self.best_validation:
                    self.best_validation = precision_val
                    is_best = True
                self.save(epoch, is_best)
            except KeyboardInterrupt:
                self.save(epoch, False)
                break

    def save(self, epoch, is_best):
        state = {
            'epoch': epoch + 1,
            'best_validation': self.best_validation,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        save_checkpoint(state, self.model_dir, is_best)

    def evaluate(self, loader_val, epoch):
        hits, misses = 0, 0
        for i, example in enumerate(loader_val):
            lc, rc, gt = [Variable(im, volatile=True).cuda(async=True) for im in example]
            gt = gt[:, self.crop: -self.crop, self.crop: -self.crop]
            out = self.model.forward(lc, rc)
            hit, miss = precision_th(gt.data, out.data, max_disp=self.max_disp, max_error=self.max_error)
            print('{}: epoch {} {}/{} {}%'.format('valid', epoch, hit, (hit + miss), 100 * (hit / (hit + miss))))
            hits, misses = hits + hit, misses + miss
        return hits, misses
