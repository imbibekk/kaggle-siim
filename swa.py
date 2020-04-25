
import os
import argparse
import pprint

import numpy as np
import torch
import torch.nn.functional as F

from datasets import get_dataloader
from models import get_model
import utils.swa as swa
from utils import checkpoint
from pathlib import Path
import pandas as pd


def get_checkpoints(args):
    checkpoint_dir = args.log_dir + '/' + args.model_name + f'/fold_{args.fold}' + '/checkpoint'
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir) if checkpoint.endswith('.pth')]

    checkpoints = [os.path.join(checkpoint_dir, ckpt) for ckpt in sorted(checkpoints)]
    checkpoints = checkpoints[-args.num_checkpoint:]
    return checkpoints


def run(args):
    df = pd.read_csv(args.df_path)
    df_train = df[df['fold']!=args.fold]

    model = get_model(args).cuda()  
    dataloader = get_dataloader(args.data_dir, df_train, 'train', args.positive_ratio, args.batch_size) 
    checkpoints = get_checkpoints(args)

    checkpoint.load_checkpoint(args, model, None, checkpoint=checkpoints[0])   # args, model, ckpt_name, checkpoint=None, optimizer=None
    for i, ckpt in enumerate(checkpoints[1:]):
        print(i, ckpt)
        model2 = get_model(args).cuda()
        last_epoch, _ = checkpoint.load_checkpoint(args, model2, None, checkpoint=ckpt)
        if args.ema is None:
            swa.moving_average(model, model2, 1. / (i + 2))
        else:
            swa.moving_average(model, model2, args.ema)

    with torch.no_grad():
        swa.bn_update(dataloader, model)

    if args.ema is not None:
        output_name = f'model_ema_{len(checkpoints)}'
    else:
        output_name = f'model_swa_{len(checkpoints)}'

    print('save {}'.format(output_name))

    checkpoint.save_checkpoint(args, model, None, 0, 0,
                                     name=output_name,
                                     weights_dict={'state_dict': model.state_dict()})


def parse_args():
    parser = argparse.ArgumentParser(description='Kaggle SIIM Competition')
    parser.add_argument('--gpu', type=int, default=0, 
                    help='Choose GPU to use. This only support single GPU')
    parser.add_argument('--data_dir', default='./data/train',
                    help='datasest directory')
    parser.add_argument('--df_path', default='./data/train_folds_5.csv',
                    help='df_path')
    parser.add_argument('--fold', type=int, default=0,
                    help='which fold to use for training')
    parser.add_argument('--positive_ratio', type=tuple, default=0.4, 
                    help='postive ratio rate for sliding sampling')
    parser.add_argument('--batch_size', type=int, default=16, 
                    help='batch size')
    parser.add_argument('--model_name', type=str, default='u_res34',
                    help='model_name as exp_name')
    parser.add_argument('--encoder_name', type=str, default='resnet34', 
                    help='which encode to use for model')
    parser.add_argument('--decoder_name', type=str, default='Unet',
                    help='type of decoder to use for model')
    parser.add_argument('--num_class', type=int, default=1,
                    help='number of classes')
    parser.add_argument('--ema', type=float, default=0.33,
                    help='ema for exponential moving average')
    parser.add_argument('--num_checkpoint', type=int, default=3,
                    help='number of snapshots to use for SWA')
    parser.add_argument('--log_dir', type=str, default='runs', 
                    help='logging directory')                

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']= f'{args.gpu}'
    pprint.PrettyPrinter(indent=2).pprint(args)
    run(args)

    print('success!')


if __name__ == '__main__':
    main()