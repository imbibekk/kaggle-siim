from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import pandas as pd
import numpy as np
import tqdm
import math
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import get_dataloader
from models import get_model, Model
from utils import checkpoint
from utils.utils import Logger, seed_everything, mask2rle


def logit_to_probability(logit_mask):
    probability_mask  = torch.sigmoid(logit_mask )
    return probability_mask


def get_models(args, ckpts):
    models = []
    for ckpt in ckpts:
        model = get_model(args).cuda()
        _= checkpoint.load_checkpoint(args, model, checkpoint=ckpt)
        model.eval()
        models.append(model)
    return models


def inference_submit(model, dataloader, augment):

    test_ids = [] 
    test_mask_probs  = [] 

    model.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, total=len(dataloader)):
            images = data['image'].cuda()
            image_id = data['image_id']
            
            num_augment=0
            probability_mask=0
            
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

            probability_mask = probability_mask.detach().cpu().numpy()

            test_ids.extend(image_id)
            test_mask_probs.append(probability_mask)

        test_mask_probs = np.concatenate(test_mask_probs)
        return test_ids, test_mask_probs


def submit(args, log):
    df = pd.read_csv(args.df_path)
    print(df.head())

    if args.ensemble:
        ckpts = args.initial_ckpt.split(',')
        models = get_models(args, ckpts)
        model = Model(models)
    else:
        model = get_model(args).cuda()
        last_epoch, step  = checkpoint.load_checkpoint(args, model, checkpoint=args.initial_ckpt)
        log.write(f'Loaded checkpoint from {args.initial_ckpt} @ {last_epoch}\n')

    dataloader = get_dataloader(args.data_dir, df, 'test', args.positive_ratio, args.batch_size)   
    seed_everything()
    
    # inference
    test_ids, mask_predictions = inference_submit(model, dataloader, args.tta_augment)
    assert len(test_ids) == mask_predictions.shape[0]

    rle_predictions = {}
    EMPTY = '-1'
    FINAL_SIZE = (1024, 1024)
    top_score_threshold = 0.75
    min_contour_area = 2000
    bot_score_threshold = 0.3

    for i, image_id in tqdm.tqdm(enumerate(test_ids), total=len(test_ids)):    
        predicted = mask_predictions[i].T
        classification_mask = predicted > top_score_threshold
        if np.sum(classification_mask) < min_contour_area:
            rle_predictions[image_id] = EMPTY
        else:
            mask_p = predicted.copy()
            mask_b = np.array(mask_p > bot_score_threshold).astype(np.uint8)
            mask = mask_b * 255
            rle_predictions[image_id] = mask2rle(mask, FINAL_SIZE)

    submission = pd.DataFrame(
        {
            'ImageId': list(rle_predictions.keys()),
            'EncodedPixels': list(rle_predictions.values())
        }
    )
    submission.loc[submission['EncodedPixels'] == '', 'EncodedPixels'] = EMPTY
    submission.to_csv(args.sub_name, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle SIIM Competition')
    parser.add_argument('--gpu', type=int, default=0, 
                    help='Choose GPU to use. This only support single GPU')
    parser.add_argument('--data_dir', default='./data/stage_2',
                    help='datasest directory')
    parser.add_argument('--df_path', default='./data/stage_2_sample_submission.csv',
                    help='df_path')
    parser.add_argument('--fold', type=int, default=0,
                    help='which fold to use for training')
    parser.add_argument('--ensemble', type=bool, default=False,
                    help='whether to use ensemble for submission')
    parser.add_argument('--positive_ratio', type=tuple, default=0.8, 
                    help='postive ratio rate for sliding sampling')
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
    parser.add_argument('--tta_augment', type=list, default=['null', 'flip_lr'], 
                    help='test time augmentation')
    parser.add_argument('--sub_name', type=str, default='submission.csv', 
                    help='output name for submission file')
    parser.add_argument('--log_dir', type=str, default='runs', 
                    help='logging directory')                

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= f'{args.gpu}'

    log = Logger()
    log.open(args.log_dir + '/' + args.model_name + f'/fold_{args.fold}' + '/submit_log.txt', mode='a')
    log.write('*'*30)
    log.write('\n')
    log.write('Logging arguments!!\n')
    log.write('*'*30)
    log.write('\n')
    for arg, value in sorted(vars(args).items()):
        log.write(f'{arg}: {value}\n')
    log.write('*'*30)
    log.write('\n')

    submit(args, log)
                     
    