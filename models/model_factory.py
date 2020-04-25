from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential
import segmentation_models_pytorch as smp


class Model:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        logits_mask = []
        with torch.no_grad():
            for m in self.models:
                logits = m(x)
                logits_mask.append(logits)
        logits_mask = torch.stack(logits_mask)
        logits_mask = torch.mean(logits_mask, dim=0)
        return logits_mask

    def eval(self):
        pass


class SIIM_Model(nn.Module):
    def __init__(self, encoder_name, decoder_name, num_class):
        super(SIIM_Model, self).__init__()
        
        if decoder_name == 'Unet':
            self.model = smp.Unet(encoder_name, classes=num_class, )
        elif decoder_name == 'FPN':
            self.model = smp.FPN(encoder_name, classes=num_class)
        else:
            raise NotImplemented

    def forward(self, x):
        logit = self.model(x)
        return logit

def get_model(args):
    return SIIM_Model(encoder_name=args.encoder_name, decoder_name=args.decoder_name, num_class=args.num_class)


if __name__ == '__main__':
   
    import argparse
    
    parser = argparse.ArgumentParser(description='Kaggle SIIM Competition')
    parser.add_argument('--encoder_name', type=str, default='resnet34', 
                    help='which encode to use for model')
    parser.add_argument('--decoder_name', type=str, default='Unet', 
                    help='which decoder to use for model')
    parser.add_argument('--num_class', type=int, default=1,
                    help='number of classes')               
    args = parser.parse_args()

    model = get_model(args)
    
    dump_inp = torch.randn((16, 3, 512, 512))
    outs = model(dump_inp)
    print(outs.shape)