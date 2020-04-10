from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import pydicom
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

from utils.utils import rle2mask


def get_mask(encode, height, width):
    if encode == [] or encode == [' -1']:
        return rle2mask('-1', height, width)
    mask = rle2mask(encode[0], height, width)
    for e in encode[1:]:
        mask += rle2mask(e, height, width)
    return mask.T


def convert_dicom_to_png(args):
    if args.mode == 'train':
        dicom_files = sorted(glob(f'{args.data_path}/siim-original/dicom-images-train/*/*/*.dcm'))
        for f in ['images', 'masks'] : os.makedirs(args.data_path +f'/{args.mode}/'+ f, exist_ok=True)
    elif args.mode == 'test':
        dicom_files = sorted(glob(f'{args.data_path}/siim-original/dicom-images-test/*/*/*.dcm'))
        os.makedirs(args.data_path + f'/{args.mode}/images', exist_ok=True)
    elif args.mode == 'stage_2':
        dicom_files = sorted(glob(f'{args.data_path}/stage_2_images/*.dcm'))
        os.makedirs(args.data_path + f'/{args.mode}/images', exist_ok=True)
    else:
        raise NotImplemented

    print(f'Number of dicom files in {args.mode}: {len(dicom_files)}')

    rle = pd.read_csv(args.df_path) 
    print(rle.head())
    
    for file_name in tqdm(dicom_files):
        img = pydicom.read_file(file_name).pixel_array
        name = file_name.split('/')[-1][:-4]
        if args.mode == 'train':
            rle_encoded = list(rle.loc[rle['ImageId'] == name, ' EncodedPixels'].values)   
            mask = get_mask(rle_encoded, img.shape[0], img.shape[1])     
            cv2.imwrite(f'{args.data_path}/{args.mode}/masks/{name}.png', mask)
        
        cv2.imwrite(f'{args.data_path}/{args.mode}/images/{name}.png', img)
            


def parse_args():
    parser = argparse.ArgumentParser(description='Kaggle SIIM Competition')
    parser.add_argument('--data_path', default='./data',
                    help='datasest directory containing dicom files')
    parser.add_argument('--df_path', default='./data/train-rle.csv',
                    help='df_path')                 
    parser.add_argument('--mode', type=str, default='stage_2',
                    help='mode: train or test or stage2')
    return parser.parse_args()

def main():
    import warnings
    warnings.filterwarnings("ignore")
    args = parse_args()
    convert_dicom_to_png(args)


if __name__ == '__main__':
    main()
    
