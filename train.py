from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
import tqdm
import copy

import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import get_dataloader
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
from utils.mask_binarizers import TripletMaskBinarization
import utils
from utils import checkpoint
from utils.metrics import dice_metric
from utils.utils import Logger, seed_everything, update_avg, accumulate


def evaluate_single_epoch(args, model, dataloader, criterion, binarizer_fn):

    model.eval()
    thresholds = binarizer_fn.thresholds
    metrics = defaultdict(float)
    curr_loss_avg = 0
    valid_preds, valid_targets = [], [] 
    
    tbar = tqdm.tqdm(dataloader,  total=len(dataloader))
    with torch.no_grad():
        for batch_idx, data in enumerate(tbar):
            images = data['image'].cuda()
            masks = data['mask'].cuda()

            masks_logits = model(images)
            masks_prob = torch.sigmoid(masks_logits)

            valid_preds.append(masks_prob.detach().cpu())
            valid_targets.append(masks.detach().cpu())
            
            loss = criterion(masks_logits, masks)
            curr_loss_avg = update_avg(curr_loss_avg, loss, batch_idx)        
            tbar.set_description('loss: {:.4}'.format(curr_loss_avg.item()))

        valid_preds = torch.cat(valid_preds)
        valid_targets = torch.cat(valid_targets)

        mask_generator = binarizer_fn.transform(valid_preds)
        for current_thr, current_mask in zip(thresholds, mask_generator):
                current_metric = dice_metric(current_mask, valid_targets, per_image=True).item()
                current_thr = tuple(current_thr)
                metrics[current_thr] = current_metric  

        best_threshold = max(metrics, key=metrics.get)
        best_metric = metrics[best_threshold]
            
    return curr_loss_avg.item(), best_metric, best_threshold


def train_single_epoch(args, model, ema_model, dataloader, criterion, optimizer, epoch, use_amp=False):
    
    model.train()
    curr_loss_avg = 0
    tbar = tqdm.tqdm(dataloader,  total=len(dataloader))
    for batch_idx, data in enumerate(tbar):
        images = data['image'].cuda()
        masks = data['mask'].cuda()

        masks_logits = model(images)  
        loss = criterion(masks_logits, masks)  
        
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if not use_amp:
            loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        if args.ema:
            if epoch >= args.ema_start:
                accumulate(ema_model, model, decay=args.ema_decay)
            else:
                accumulate(ema_model, model, decay=0)

        curr_loss_avg = update_avg(curr_loss_avg, loss, batch_idx)        
        tbar.set_description('loss: %.5f, lr: %.6f' % (curr_loss_avg.item(), optimizer.param_groups[0]['lr']))
    return curr_loss_avg.item()

        
def train(args, log, model, dataloaders, criterion, optimizer, scheduler, binarizer_fn, start_epoch):
    num_epochs = args.num_epochs

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    if args.ema:
        ema_model = copy.deepcopy(model)
        ema_model.cuda()
    else:
        ema_model = None

    log.write('Start Training...!!\n')

    patience = 0.0
    best_val_loss = 10.0
    best_dice_score = 0.0
    for epoch in range(start_epoch, num_epochs):

        # train phase
        train_loss = train_single_epoch(args, model, ema_model, dataloaders['train'], criterion, optimizer, epoch)
        
        # valid phase
        if args.ema:
            val_loss, val_dice, best_threshold = evaluate_single_epoch(args, ema_model, dataloaders['val'], criterion, binarizer_fn)
        else:
            val_loss, val_dice, best_threshold = evaluate_single_epoch(args, model, dataloaders['val'], criterion, binarizer_fn)
        
        log_write = f'Epoch: {epoch} | Train_loss: {train_loss:.5f} | Val loss: {val_loss:.5f} | val dice: {val_dice:.5f} @ {best_threshold}'
        log.write(log_write)
        log.write('\n')
            
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_dice)
        else:
            scheduler.step()
        
        # save metric checkpoint
        name = 'metric' 
        if args.ema:
            checkpoint.save_checkpoint(args, ema_model, optimizer, epoch=epoch, metric_score=val_dice, step=0, keep=args.ckpt_keep, name=name)  
        else:
            checkpoint.save_checkpoint(args, model, optimizer, epoch=epoch, metric_score=val_dice, step=0, keep=args.ckpt_keep, name=name)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if val_dice > best_dice_score:
            patience = 0.0
            best_dice_score = val_dice
        else:
            patience += 1
            if patience ==10:
                log.write(f'Early Stopping....@ {epoch} epoch for patience @ {patience}\n')
                log.write(f'Best Loss: {best_val_loss} | Best Dice: {best_dice_score}\n')        
                break
        
        log.write(f'Best Loss: {best_val_loss} | Best Dice: {best_dice_score}\n')

        
def run(args, log):

    df = pd.read_csv(args.df_path)
    df_train = df[df['fold']!=args.fold]
    df_valid = df[df['fold']==args.fold]
    dfs = {}
    dfs['train'] = df_train
    dfs['val'] = df_valid
    
    model = get_model(args).cuda()

    criterion = get_loss(args)
    optimizer = get_optimizer(args, model.parameters())

    triplets = [[0.75, 1000, 0.3], [0.75, 1000, 0.4], [0.75, 2000, 0.3], [0.75, 2000, 0.4], [0.6, 2000, 0.3], [0.6, 2000, 0.4], [0.6, 3000, 0.3], [0.6, 3000, 0.4]]
    binarizer_fn = TripletMaskBinarization(triplets)
    
    if args.initial_ckpt is not None:
        last_epoch, step = checkpoint.load_checkpoint(args, model, checkpoint=args.initial_ckpt)
        log.write(f'Resume training from {args.initial_ckpt} @ {last_epoch}\n')
    else:
        last_epoch, step = -1, -1

    dataloaders = {mode:get_dataloader(args.data_dir, dfs[mode], mode, args.positive_ratio, args.batch_size) for mode in ['train', 'val']}   

    scheduler = get_scheduler(args, optimizer, -1, dataloaders['train'])

    seed_everything()

    train(args, log, model, dataloaders, criterion, optimizer, scheduler, binarizer_fn, last_epoch+1)


def parse_args():
    parser = argparse.ArgumentParser(description='Kaggle SIIM Competition')
    parser.add_argument('--gpu', type=int, default=0, 
                    help='Choose GPU to use. This only support single GPU')
    parser.add_argument('--data_dir', default='./data/train',
                    help='datasest directory')
    parser.add_argument('--df_path', default='./data/train_folds_5.csv',
                    help='df_path')                 
    parser.add_argument('--fold', type=int, default=1,
                    help='which fold to use for training')
    parser.add_argument('--model_name', type=str, default='u_res34',
                    help='model_name as exp_name')                 
    parser.add_argument('--batch_size', type=int, default=16, 
                    help='batch size')
    parser.add_argument('--num_epochs', type=int, default=50, 
                    help='num of epochs to train')
    parser.add_argument('--positive_ratio', type=float, default=0.6, 
                    help='postive ratio rate for sliding sampling')
    parser.add_argument('--loss_weights', type=dict, default={'bce': 1, 'dice': 1, 'focal': 1},
                    help='name of optimizer to use')
    parser.add_argument('--grad_clip', type=float, default=0.1,
                    help='clipping value for gradient')
    parser.add_argument('--ema',type=bool, default=False,
                    help='whether to use ema or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--ema_start', type=int, default=5)
    parser.add_argument('--optimizer_name', type=str, default='adam',
                    help='name of optimizer to use')
    parser.add_argument('--scheduler_name', type=str, default='cosine',
                    help='learning rate scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-4,
                    help='minimum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.000005,
                    help='weight decay for optimizer')   
    parser.add_argument('--encoder_name', type=str, default='resnet34', 
                    help='which encode to use for model')
    parser.add_argument('--decoder_name', type=str, default='Unet',
                    help='type of decoder to use for model')
    parser.add_argument('--num_class', type=int, default=1,
                    help='number of classes')                 
    parser.add_argument('--initial_ckpt', type=str, default=None,
                    help='inital checkpoint to resume training')
    parser.add_argument('--ckpt_keep', type=int, default=5,
                    help='how many checkpoints to save')                              
    parser.add_argument('--log_dir', type=str, default='runs', 
                    help='logging directory')                

    return parser.parse_args()

def main():
    import warnings
    warnings.filterwarnings("ignore")
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= f'{args.gpu}'
    utils.prepare_train_directories(args)
    
    log = Logger()
    log.open(args.log_dir + '/' + args.model_name + f'/fold_{args.fold}' + '/train_log.txt', mode='a')
    log.write('*'*30)
    log.write('\n')
    log.write('Logging arguments!!\n')
    log.write('*'*30)
    log.write('\n')
    for arg, value in sorted(vars(args).items()):
        log.write(f'{arg}: {value}\n')
    log.write('*'*30)
    log.write('\n')

    run(args, log)
    print('success!')

if __name__ == '__main__':
    main()


