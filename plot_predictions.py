from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from utils.mask_binarizers import TripletMaskBinarization
from utils.metrics import dice_metric
from utils.utils import Logger, read_list_from_file, read_pickle_from_file


def draw_truth_overlay(image, component, alpha=0.5):
    component = component*255
    overlay   = image.astype(np.float32)
    overlay[:,:,2] += component*alpha
    overlay = np.clip(overlay,0,255)
    overlay = overlay.astype(np.uint8)
    return overlay


def draw_predict_overlay(image, component, alpha=0.5):
    component = component*255
    overlay   = image.astype(np.float32)
    overlay[:,:,1] += component*alpha
    overlay = np.clip(overlay,0,255)
    overlay = overlay.astype(np.uint8)
    return overlay


def draw_input_overlay(image):
    overlay = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    return overlay


def visualize(plot_dict, image_id):
    """PLot images in one row."""
    n = len(plot_dict)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(plot_dict.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.suptitle(image_id)
    plt.show()


def plot_predictions(args, log):
    out_dir = args.log_dir + f'/{args.model_name}/fold_{args.fold}/submit'
    log.write(f'loading predictions from {out_dir}\n')

    image_ids = read_list_from_file(out_dir + '/image_id.txt')
    mask_probs = read_pickle_from_file(out_dir + '/mask_probs.pickle')
    mask_truths = read_pickle_from_file(out_dir + '/mask_truth.pickle')
    
    mask_probs = torch.from_numpy(mask_probs)
    mask_truths = torch.from_numpy(mask_truths)

    best_thresholds = [(0.75, 2000, 0.3)]
    binarizer_fn = TripletMaskBinarization(best_thresholds)
    mask_generator = binarizer_fn.transform(mask_probs)

    for current_thr, pred_mask in zip(best_thresholds, mask_generator):
            dice = dice_metric(pred_mask, mask_truths, per_image=True).item()

    mask_truths = mask_truths.numpy()
    mask_probs = pred_mask.numpy()

    for image_id, mask_preds, mask_truth in zip(image_ids, mask_probs, mask_truths):
        
        image = cv2.imread(os.path.join(args.data_dir, 'images', image_id ), 1)
        image = draw_input_overlay(image)

        image = cv2.resize(image, (mask_preds.shape[1], mask_preds.shape[2]))
        mask_truth = np.transpose(mask_truth, (1,2,0))
        mask_preds = np.transpose(mask_preds, (1,2,0))
        
        mask_truth = np.squeeze(mask_truth, axis=-1)
        mask_preds = np.squeeze(mask_preds, axis=-1)

        image_truth = draw_truth_overlay(image.copy(), mask_truth, 0.5)
        image_preds = draw_predict_overlay(image.copy(), mask_preds, 0.5)

        visualize({'ground_truth':image_truth, 'prediction':image_preds}, image_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle SIIM Competition')
    parser.add_argument('--data_dir', default='./data/train',
                    help='datasest directory')
    parser.add_argument('--save_dir',type=bool, default=True,
                    help='saving directory for predictions')
    parser.add_argument('--fold', type=int, default=0,
                    help='which fold to use for training')
    parser.add_argument('--model_name', type=str, default='FPN_effb4',
                    help='model_name as exp_name')
    parser.add_argument('--log_dir', type=str, default='runs', 
                    help='logging directory')                

    args = parser.parse_args()

    log = Logger()
    log.open(args.log_dir + '/' + args.model_name + f'/fold_{args.fold}' + '/plot_log.txt', mode='a')
    log.write('*'*30)
    log.write('\n')
    log.write('Logging arguments!!\n')
    log.write('*'*30)
    log.write('\n')
    for arg, value in sorted(vars(args).items()):
        log.write(f'{arg}: {value}\n')
    log.write('*'*30)
    log.write('\n')

    plot_predictions(args, log)
                     




