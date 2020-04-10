from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import torch


def get_last_checkpoint(checkpoint_dir, name):

    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir)
                   if checkpoint.startswith(name) and checkpoint.endswith('.pth')]
    if checkpoints:
        return os.path.join(checkpoint_dir, list(sorted(checkpoints))[-1])
    return None


def get_initial_checkpoint(args, name='best_loss'):
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoint')
    return get_last_checkpoint(checkpoint_dir, name)


def get_checkpoint(args, name):
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoint')
    return os.path.join(checkpoint_dir, name)


def remove_old_checkpoint(checkpoint_dir, keep, name):
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir)
                   if name in str(checkpoint) and checkpoint.endswith('.pth')]
    checkpoints.sort(key = lambda x: float(x.split('_')[0]))
    for checkpoint in checkpoints[:-keep]:
        os.remove(os.path.join(checkpoint_dir, checkpoint))


def copy_last_n_checkpoints(args, n, name):
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoint')
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir)
                   if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
    checkpoints = sorted(checkpoints)
    for i, checkpoint in enumerate(checkpoints[-n:]):
        shutil.copyfile(os.path.join(checkpoint_dir, checkpoint),
                        os.path.join(checkpoint_dir, name.format(i)))


def load_checkpoint(args, model, ckpt_name=None, checkpoint=None, optimizer=None):
    if checkpoint is None:
        checkpoint_dir = os.path.join(args.log_dir, args.model_name,  f'fold_{args.fold}', 'checkpoint', ckpt_name)
        checkpoint = torch.load(checkpoint_dir)
    else:
        checkpoint = torch.load(checkpoint)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

    step = checkpoint['step'] if 'step' in checkpoint else -1
    last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1
    return last_epoch, step


def save_checkpoint(args, model, optimizer, epoch, metric_score=0, step=0, keep=10, weights_dict=None, name=None):
    checkpoint_dir = os.path.join(args.log_dir, args.model_name,  f'fold_{args.fold}', 'checkpoint')

    if name:
        if metric_score:
            checkpoint_path = os.path.join(checkpoint_dir, '{}_{}.pth'.format(metric_score, name))
        else:
            checkpoint_path = os.path.join(checkpoint_dir, '{}_{}.pth'.format(epoch, name))
    else:
        checkpoint_path = os.path.join(checkpoint_dir, '{}_epoch.pth'.format(epoch))

    state_dict = {}
    for key, value in model.state_dict().items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        state_dict[key] = value

    if weights_dict is None:
        weights_dict = {
          'state_dict': state_dict,
          'optimizer_dict' : optimizer.state_dict(),
          'epoch' : epoch,
          'step' : step,
        }
    torch.save(weights_dict, checkpoint_path)

    if keep is not None and keep > 0:
        remove_old_checkpoint(checkpoint_dir, keep, name)


if __name__ == '__main__':
    print('this is working!!')
    name = 'best_train_loss'
    keep = 15
    checkpoint_dir = '../runs/checkpoint'
    remove_old_checkpoint(checkpoint_dir, keep, name)