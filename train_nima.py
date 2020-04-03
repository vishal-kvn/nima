import importlib.util
import math
import os.path
from os import path
import time

from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.callbacks.tracker import *
torch.backends.cudnn.benchmark = True
import numpy as np
import pandas as pd
from torch import nn

# import pdb
# pdb.set_trace()

# tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
tfms = ([
    flip_lr(p=0.5),
    brightness(change=(0.4,0.6)),
    contrast(scale=(0.7,1.3))
], [])

df = pd.read_csv('./labels_sample.csv')
scores_map = dict(zip(df.image_name, df.tags))

func = lambda o: int((o.split('/')[2]).split('.')[0])

labels = list(scores_map.keys())
class NimaLabelList(CategoryList):
    _processor=None
    def __init__(self, items:Iterator, classes=labels, label_delim:str=None, **kwargs):
        super().__init__(items, classes=classes, **kwargs)

    def get(self, i):
        dist = scores_map[self.items[i]]
        dist = np.array(dist.split(' '), dtype=int)
        return dist

import pdb
pdb.set_trace()
# data = (ImageList.from_csv('./', 'labels_sample.csv', folder='data', suffix='.jpg')
#         .split_by_rand_pct()
#         .label_from_func(func, label_cls=NimaLabelList)
#         .transform(tfms, size=224)
#         .databunch(bs=8))

data = ImageList.from_csv('./', 'labels_sample.csv', folder='data', suffix='.jpg')
data = data.split_by_rand_pct()
data = data.label_from_func(func, label_cls=NimaLabelList)
# data = data.label_from_df(label_delim=' ')
data = data.transform(tfms, size=224)
import pdb
pdb.set_trace()
data = data.databunch(bs=8)

x,y = next(iter(data.train_dl))
import pdb
pdb.set_trace()

# For precision and recall curves
# https://medium.com/analytics-vidhya/multi-label-classification-using-fastai-a-shallow-dive-into-fastai-data-block-api-54ea57b2c78b

data.c = 10

def emd(y, y_hat):
    cdf_y = torch.cumsum(y, dim=-1)
    cdf_y_hat = torch.cumsum(y_hat, dim=-1).double()
    power = torch.pow((cdf_y - cdf_y_hat), 2)
    emd = torch.sqrt(torch.mean(power, dim=-1))
    return torch.mean(emd)

arch  = models.mobilenet_v2
learn = cnn_learner(data, arch, pretrained=True)
learn.loss_func = emd

y_hat = learn.model(x)

