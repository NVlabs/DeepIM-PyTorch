# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.ycb_video
import datasets.ycb_object
import datasets.linemod
import datasets.panda
import datasets.background
import datasets.dex_ycb
import numpy as np

# ycb video dataset
for split in ['train', 'val', 'keyframe', 'trainval', 'debug']:
    name = 'ycb_video_{}'.format(split)
    #print name
    __sets[name] = (lambda split=split:
            datasets.YCBVideo(split))

# ycb object dataset
for split in ['train', 'test']:
    name = 'ycb_object_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.YCBObject(split))

# background dataset
for split in ['coco', 'rgbd', 'nvidia', 'table', 'isaac', 'texture', 'sunrgbd']:
    name = 'background_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.BackgroundDataset(split))

# DEX YCB dataset
for setup in ('s0', 's1', 's2', 's3'):
    for split in ('train', 'val', 'test'):
        name = 'dex_ycb_{}_{}'.format(setup, split)
        __sets[name] = (lambda setup=setup, split=split: datasets.DexYCBDataset(setup, split))

def get_dataset(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_datasets():
    """List all registered imdbs."""
    return __sets.keys()
