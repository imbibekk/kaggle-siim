import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import torch
from typing import List, Tuple
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import albumentations as A
from albumentations.pytorch import ToTensor
from torchvision.utils import make_grid


class SIIMDataset(Dataset):
    def __init__(self, df, data_dir, mode ='train', transform=None):

        if mode == 'test':
            self.image_ids = df['ImageId'].tolist()
        else:
            self.image_ids = df['fname'].tolist()
            self.non_emptiness = df['exist_labels'].values
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        if self.mode == 'test':
            image_dir = os.path.join(self.data_dir, 'images', image_id + '.png')
        else:
            image_dir = os.path.join(self.data_dir, 'images', image_id)
        image = cv2.imread(image_dir, 1)
    
        if self.mode == 'test':
            if self.transform is not None:
                augmented = self.transform(image=image)
                image = augmented['image']
            data = {}
            data['image'] = image
            data['image_id'] = image_id
            return data
        else:
            mask_dir = os.path.join(self.data_dir, 'masks', image_id) 
            mask = cv2.imread(mask_dir, 0) 
            if self.transform is not None:
                augmented = self.transform(image=image, mask= mask)
                image = augmented['image']
                mask = augmented['mask']

            data = {}
            data['image'] = image
            data['mask'] = mask
            data['image_id'] = image_id
            return data


class PneumoSampler(Sampler):
    def __init__(self, df, demand_non_empty_proba):
        assert demand_non_empty_proba > 0, 'frequensy of non-empty images must be greater then zero'
        self.positive_proba = demand_non_empty_proba
        self.folds = df.reset_index(drop=True)

        self.positive_idxs = self.folds[self.folds.exist_labels == 1].index.values
        self.negative_idxs = self.folds[self.folds.exist_labels == 0].index.values

        self.n_positive = self.positive_idxs.shape[0]
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)
        
    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative


class EmptySampler(Sampler):
    def __init__(self, data_source, positive_ratio_range, epochs):
        super().__init__(data_source)
        assert len(positive_ratio_range) == 2
        self.positive_indices = np.where(data_source.non_emptiness == 1)[0]
        self.negative_indices = np.where(data_source.non_emptiness == 0)[0]
        self.positive_ratio_range = positive_ratio_range
        self.positive_num: int = len(self.positive_indices)
        self.current_epoch: int = 0
        self.epochs: int = epochs

    @property
    def positive_ratio(self) -> float:
        np.random.seed(self.current_epoch)
        min_ratio, max_ratio = self.positive_ratio_range
        return max_ratio - (max_ratio - min_ratio) / self.epochs * self.current_epoch

    @property
    def negative_num(self) -> int:
        assert self.positive_ratio <= 1.0
        return int(self.positive_num // self.positive_ratio - self.positive_num)

    def __iter__(self):
        negative_indices = np.random.choice(self.negative_indices, size=self.negative_num)
        indices = np.random.permutation(np.hstack((negative_indices, self.positive_indices)))
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.positive_num + self.negative_num

    def set_epoch(self, epoch):
        self.current_epoch = epoch



def generate_transforms():

    train_transform = A.Compose([
                                A.HorizontalFlip(),
                                A.OneOf([
                                    A.RandomContrast(),
                                    A.RandomGamma(),
                                    A.RandomBrightness(),
                                        ], p=0.3),
                                A.OneOf([
                                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                                    A.GridDistortion(),
                                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                                        ], p=0.3),

                                A.ShiftScaleRotate(),
                                ToTensor()
                                ])
    
    
    valid_transform = A.Compose([
                                ToTensor()
                                ])
    
    
    return train_transform, valid_transform


def get_dataloader(args, df, mode):
    
    train_transform, valid_transform = generate_transforms()

    datasets = SIIMDataset(df, args.data_dir, mode, transform=train_transform if mode =='train' else valid_transform)
    
    is_train = mode =='train'
    if is_train:
        if args.sampler_name == 'EmptySampler':
            sampler = EmptySampler(datasets, [0.2,0.8], args.num_epochs)
        else:
            sampler = PneumoSampler(df, args.demand_non_empty_proba)

    dataloader = DataLoader(datasets,
                            batch_size=args.batch_size if is_train else 2*args.batch_size,
                            sampler=sampler if is_train else None, 
                            num_workers=4,
                            )
    return dataloader



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import math

    data_dir = '../data/train'
    mode = 'train'
    fold = 0
    positive_ratio_range = (0.2, 0.8)
    df = pd.read_csv('../data/train_splits.csv')
    df_train = df[df['fold']!=fold]
    print(df_train['fold'].value_counts())

    def show(img, mask):
        npimg = img.numpy()
        nmask = mask.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.imshow(np.transpose(nmask, (1,2,0)), interpolation='nearest', alpha=0.5)
        plt.show()

    loader = get_dataloader(data_dir, df_train, mode, 0.8, batch_size=16)
    for data in loader:
        images = data['image']
        masks = data['mask']
        print(images.shape, masks.shape, cls_label.shape)
        show(make_grid(images), make_grid(masks))
        break

        
        
