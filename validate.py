from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import torch
import torch.nn.functional as F

import tqdm
import pandas as pd
import numpy as np
import math
from collections import defaultdict

from datasets import get_dataloader
from models import get_model
from utils import checkpoint
from utils.mask_binarizers import TripletMaskBinarization
from utils.metrics import dice_metric
from utils.utils import Logger, seed_everything
from utils.utils import read_list_from_file, read_pickle_from_file, write_list_to_file, write_pickle_to_file


def logit_to_probability(logit_mask):
    probability_mask  = torch.sigmoid(logit_mask )
    return probability_mask


def evaluate_single_epoch(model, dataloader, augment):

    mask_probs = []
    image_ids, mask_truth = [], []
    
    model.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, total=len(dataloader)):
            images = data['image'].cuda()
            masks = data['mask'].cuda()
            image_id = data['image_id']

            num_augment=0.0
            probability_mask=0.0
            
            if 'null' in augment: #1: #  null
                logit_mask = model(images)  
                p_mask = logit_to_probability(logit_mask)

                probability_mask  += p_mask
                num_augment+=1

            if 'flip_lr' in augment:
                logit_mask = model(torch.flip(images,dims=[3]))
                p_mask = logit_to_probability(torch.flip(logit_mask,dims=[3]))

                probability_mask  += p_mask
                num_augment+=1

            if 'flip_ud' in augment:
                logit_mask = model(torch.flip(images,dims=[2]))
                p_mask = logit_to_probability(torch.flip(logit_mask,dims=[2]))

                probability_mask  += p_mask
                num_augment+=1

            if 'flip_both' in augment:
                logit_mask = model(torch.flip(images,dims=[2,3]))
                p_mask = logit_to_probability(torch.flip(logit_mask,dims=[2,3]))

                probability_mask  += p_mask
                num_augment+=1

            probability_mask  = probability_mask/num_augment
            
            mask_probs.append(probability_mask.detach().cpu().numpy())
            mask_truth.append(masks.detach().cpu().numpy())
            image_ids.extend(image_id)

        mask_probs = np.concatenate(mask_probs)
        mask_truth = np.concatenate(mask_truth)

    return image_ids, mask_probs, mask_truth


def inference(args, log):
    df = pd.read_csv(args.df_path)
    
    if args.ensemble:
        folds = args.fold.split(',')
        ckpts = args.initial_ckpt.split(',')
    else:
        folds = [args.fold]
        ckpts = [args.initial_ckpt]

    image_ids, mask_probs, mask_truth = [], [], []
    for fold, ckpt in zip(folds, ckpts):
        df_valid = df[df['fold']==int(fold)]
        
        model = get_model(args).cuda()
        _ = checkpoint.load_checkpoint(args, model, checkpoint=ckpt)
        log.write(f'Loaded checkpoint for fold {fold} from {ckpt}\n')

        dataloaders = {mode:get_dataloader(args.data_dir, df_valid, mode, args.positive_ratio, args.batch_size) for mode in ['val']}   
        seed_everything()

        # inference
        fold_image_ids, fold_mask_probs, fold_mask_truth = evaluate_single_epoch(model, dataloaders['val'], args.tta_augment)
        image_ids.append(fold_image_ids)
        mask_probs.append(fold_mask_probs)
        mask_truth.append(fold_mask_truth)
    
    image_ids = np.concatenate(image_ids)
    mask_probs = np.concatenate(mask_probs)
    mask_truth = np.concatenate(mask_truth)
    
    if args.save2numpy:
        if args.ensemble:
            out_dir = args.log_dir + f'/{args.model_name}/ensemble/submit'
        else:
            out_dir = args.log_dir + f'/{args.model_name}/fold_{args.fold}/submit'
        
        os.makedirs(out_dir, exist_ok=True)
        log.write(f'saving results @ {out_dir}\n')

        write_list_to_file(out_dir + '/image_id.txt',image_ids)
        write_pickle_to_file(out_dir + '/mask_probs.pickle', mask_probs)
        write_pickle_to_file(out_dir + '/mask_truth.pickle', mask_truth)

        if 1:
            image_id = read_list_from_file(out_dir + '/image_id.txt')
            mask_probs = read_pickle_from_file(out_dir + '/mask_probs.pickle')
            mask_truth = read_pickle_from_file(out_dir + '/mask_truth.pickle')
        
    mask_probs = torch.from_numpy(mask_probs)
    mask_truth = torch.from_numpy(mask_truth)
    log.write(f'mask_truth: {mask_truth.shape} | mask_probs: {mask_probs.shape}\n')
        
    triplets = [[0.75, 1000, 0.3], [0.75, 1000, 0.4], [0.75, 2000, 0.3], [0.75, 2000, 0.4], [0.6, 2000, 0.3], [0.6, 2000, 0.4], [0.6, 3000, 0.3], [0.6, 3000, 0.4]]
    binarizer_fn = TripletMaskBinarization(triplets)
    mask_generator = binarizer_fn.transform(mask_probs)
        
    thresholds = binarizer_fn.thresholds
    metrics = defaultdict(float)
        
    for current_thr, pred_mask in zip(thresholds, mask_generator):
        current_metric = dice_metric(pred_mask, mask_truth, per_image=True).item()
        current_thr = tuple(current_thr)
        metrics[current_thr] =  current_metric

    log.write('\n')
    for th, value in sorted(metrics.items()):
        log.write(f'{th}: {value}\n')
    log.write('\n')

    best_threshold = max(metrics, key=metrics.get)
    best_metric = metrics[best_threshold]            
    log.write(f'Validation dice : {best_metric} @ {best_threshold}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle SIIM Competition')
    parser.add_argument('--gpu', type=int, default=0, 
                    help='Choose GPU to use. This only support single GPU')
    parser.add_argument('--data_dir', default='./data/train',
                    help='datasest directory')
    parser.add_argument('--df_path', default='./data/train_folds_5.csv',
                    help='df_path')
    parser.add_argument('--save2numpy',type=bool, default=False,
                    help='whether to save validation results or not')
    parser.add_argument('--ensemble', type=bool, default=False,
                    help='whether to use ensemble for submission')
    parser.add_argument('--fold', type=str, default=0,
                    help='which fold to use for training')
    parser.add_argument('--positive_ratio', type=tuple, default=0.8, 
                    help='postive ratio rate for sliding sampling')
    parser.add_argument('--num_epochs', type=int, default=30, 
                    help='num of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, 
                    help='batch size')
    parser.add_argument('--model_name', type=str, default='FPN_effb4',
                    help='model_name as exp_name')
    parser.add_argument('--encoder_name', type=str, default='resnet34', 
                    help='which encode to use for model')
    parser.add_argument('--decoder_name', type=str, default='Unet',
                    help='type of decoder to use for model')
    parser.add_argument('--num_class', type=int, default=1,
                    help='number of classes')
    parser.add_argument('--initial_ckpt', type=str, default=None,
                    help='inital checkpoint to resume training')
    parser.add_argument('--tta_augment', type=list, default=['null'], 
                    help='test time augmentation')
    parser.add_argument('--log_dir', type=str, default='runs', 
                    help='logging directory')                

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= f'{args.gpu}'

    log = Logger()
    if args.ensemble:
        os.makedirs(args.log_dir + f'/{args.model_name}/ensemble', exist_ok=True)
        log.open(args.log_dir + '/' + args.model_name + '/ensemble' + '/ensemble.txt', mode='a')
    else:
        log.open(args.log_dir + '/' + args.model_name + f'/fold_{args.fold}' + '/validate_log.txt', mode='a')
    log.write('*'*30)
    log.write('\n')
    log.write('Logging arguments!!\n')
    log.write('*'*30)
    log.write('\n')
    for arg, value in sorted(vars(args).items()):
        log.write(f'{arg}: {value}\n')
    log.write('*'*30)
    log.write('\n')

    inference(args, log)
                     
    