from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler


def reduce_lr_on_plateau(optimizer, last_epoch, mode='max', factor=0.1, patience=2,
                         threshold=0.0000001, threshold_mode='rel', cooldown=0, min_lr=0.0000001, **_):
  return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                        threshold=threshold, threshold_mode=threshold_mode,
                                        cooldown=cooldown, min_lr=min_lr)


def cosine(optimizer, last_epoch, T_max=8, eta_min=0.00001, **_):
  print('cosine annealing, T_max: {}, eta_min: {}, last_epoch: {}'.format(T_max, eta_min, last_epoch))
  return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min,
                                        last_epoch=last_epoch)


def get_scheduler(args, optimizer, last_epoch, train_loader):
        assert args.scheduler_name in ['cosine', 'reduce_lr_on_plateau']
        if args.scheduler_name == 'cosine':
            return cosine(optimizer, last_epoch)
        if args.scheduler_name == 'reduce_lr_on_plateau':
            return reduce_lr_on_plateau(optimizer, last_epoch)
  
  