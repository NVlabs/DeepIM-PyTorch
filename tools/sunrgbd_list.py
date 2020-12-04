# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os, sys
import glob
try:
    import cPickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle as cPickle

comotion = '../data/SUNRGBD'
filename = os.path.join(comotion, '**/depth')
print('list files in %s' % (filename))
files = glob.glob(filename, recursive=True)

f = open('../data/sunrgbd.txt', 'w')
filenames = []
for i in range(len(files)):
    filename = files[i]
    f.write('%s\n' % (filename[8:]))
    filenames.append(filename[8:])
f.close()

cache_file = '../data/sunrgbd.pkl'
with open(cache_file, 'wb') as fid:
    cPickle.dump(filenames, fid, cPickle.HIGHEST_PROTOCOL)
print('wrote filenames to {}'.format(cache_file))
