
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import re
import numpy as np
import builtins
import pickle
import random
import torch


def prepare_train_directories(args):
  out_dir = args.log_dir
  os.makedirs(os.path.join(out_dir, args.model_name, f'fold_{args.fold}', 'checkpoint'), exist_ok=True)


def update_avg(curr_avg, val, idx):
    return (curr_avg * idx + val) / (idx + 1)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass



def mask2rle(img, size):
    width, height = size
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return ' '.join(rle)



def rle2mask(rle, height, width):
    mask= np.zeros(width* height)
    if rle == ' -1' or rle == '-1':
        return mask.reshape(width,height)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)
    


def read_pickle_from_file(pickle_file):
    with open(pickle_file,'rb') as f:
        x = pickle.load(f)
    return x

def write_pickle_to_file(pickle_file, x):
    with open(pickle_file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)


def write_list_to_file(list_file, strings):
    with open(list_file, 'w') as f:
        for s in strings:
            f.write('%s\n'%str(s))
    pass


def read_list_from_file(list_file, comment='#'):
    with open(list_file) as f:
        lines  = f.readlines()

    strings=[]
    for line in lines:
        if comment is not None:
            s = line.split(comment, 1)[0].strip()
        else:
            s = line.strip()

        if s != '':
            strings.append(s)
    return strings

