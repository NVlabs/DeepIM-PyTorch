# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.utils.data as data

import os, math
import sys
import os.path as osp
from os.path import *
import numpy as np
import numpy.random as npr
import cv2
import scipy.io
import copy
import glob
try:
    import cPickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle as cPickle

import datasets
from fcn.config import cfg
from utils.blob import pad_im, chromatic_transform, add_noise, add_noise_cuda
from transforms3d.quaternions import mat2quat, quat2mat
from utils.se3 import *
from utils.pose_error import *
from datasets.ycb_video import YCBVideo


class YCBSelfSupervision(YCBVideo):
    def __init__(self, image_set, ycb_self_supervision_path = None):

        self._name = 'ycb_self_supervision_' + image_set
        self._image_set = image_set
        self._ycb_self_supervision_path = self._get_default_path() if ycb_self_supervision_path is None \
                            else ycb_self_supervision_path
        self._data_path = os.path.join(self._ycb_self_supervision_path, 'data')
        self._model_path = os.path.join(datasets.ROOT_DIR, 'data', 'models')

        # objects
        self._classes_all = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
        self._num_classes_all = len(self._classes_all)
        self._class_colors_all = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
                              (192, 0, 0), (0, 192, 0), (0, 0, 192)]
        self._symmetry_all = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        self._extents_all = self._load_object_extents()

        # camera intrinsics
        self._width = cfg.TRAIN.SYN_WIDTH
        self._height = cfg.TRAIN.SYN_HEIGHT
        self._depth_factor = 1000.0
        self._intrinsic_matrix = np.array([[616.3653,    0.,      310.25882],
                                           [  0.,      616.20294, 236.59981],
                                           [  0.,        0.,        1.     ]])

        if self._width == 1280:
            self._intrinsic_matrix = np.array([[599.48681641,   0.,         639.84338379],
                                               [  0.,         599.24389648, 366.09042358],
                                               [  0.,           0.,           1.        ]])

        # select a subset of classes
        self._classes = [self._classes_all[i] for i in cfg.TRAIN.CLASSES]
        self._num_classes = len(self._classes)
        self._class_colors = [self._class_colors_all[i] for i in cfg.TRAIN.CLASSES]
        self._symmetry = self._symmetry_all[cfg.TRAIN.CLASSES]
        self._extents = self._extents_all[cfg.TRAIN.CLASSES]
        self._points, self._points_all, self._point_blob = self._load_object_points()

        # 3D model paths
        self.model_mesh_paths = ['{}/{}/textured_simple.obj'.format(self._model_path, cls) for cls in self._classes_all]
        self.model_texture_paths = ['{}/{}/texture_map.png'.format(self._model_path, cls) for cls in self._classes_all]
        self.model_colors = [np.array(self._class_colors_all[i]) / 255.0 for i in range(len(self._classes_all))]

        self.model_mesh_paths_target = ['{}/{}/textured_simple.obj'.format(self._model_path, cls) for cls in self._classes]
        self.model_texture_paths_target = ['{}/{}/texture_map.png'.format(self._model_path, cls) for cls in self._classes]
        self.model_colors_target = [np.array(self._class_colors_all[i]) / 255.0 for i in cfg.TRAIN.CLASSES]

        self._class_to_ind = dict(zip(self._classes, range(self._num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index(image_set)
        self._pixel_mean = cfg.PIXEL_MEANS / 255.0

        if cfg.MODE == 'TRAIN' and cfg.TRAIN.SYNTHESIZE:
            self._size = len(self._image_index) * (cfg.TRAIN.SYN_RATIO+1)
        else:
            self._size = len(self._image_index)

        self._roidb = self.gt_roidb()
        self._build_uniform_poses()

        # for evaluation
        self._correct_poses = np.zeros((cfg.TEST.ITERNUM+1, 3), dtype=np.float32)
        self._total_poses = 0

        assert os.path.exists(self._ycb_self_supervision_path), \
                'ycb_self_supervision path does not exist: {}'.format(self._ycb_self_supervision_path)
        assert os.path.exists(self._data_path), \
                'Data path does not exist: {}'.format(self._data_path)
        assert os.path.exists(self._model_path), \
                'Model path does not exist: {}'.format(self._model_path)


    def _get_default_path(self):
        """
        Return the default path where ycb_self_supervision is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'YCB_Self_Supervision')


    def _load_image_set_index(self, image_set):
        """
        Load the indexes of images in the data folder
        """

        image_set_file = os.path.join(self._ycb_self_supervision_path, image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        subdirs = []
        with open(image_set_file) as f:
            for x in f.readlines():
                subdirs.append(x.rstrip('\n'))

        image_index = []
        for i in range(len(subdirs)):
            subdir = subdirs[i]
            folder = osp.join(self._data_path, subdir)
            filename = os.path.join(folder, '*.mat')
            files = glob.glob(filename)
            print(subdir, len(files))
            for k in range(len(files)):
                filename = files[k]
                head, name = os.path.split(filename)
                index = subdir + '/' + name[:-9]
                image_index.append(index)

        print('=======================================================')
        print('%d image in %s' % (len(image_index), self._data_path))
        print('=======================================================')
        return image_index


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        gt_roidb = [self._load_ycb_self_supervision_annotation(index)
                    for index in self._image_index]

        return gt_roidb


    def _load_ycb_self_supervision_annotation(self, index):
        """
        Load class name and meta data
        """
        # image path
        image_path = self.image_path_from_index(index)

        # depth path
        depth_path = self.depth_path_from_index(index)

        # metadata path
        metadata_path = self.metadata_path_from_index(index)

        is_syn = 0
        video_id = ''
        image_id = ''
        
        return {'image': image_path,
                'depth': depth_path,
                'meta_data': metadata_path,
                'video_id': video_id,
                'image_id': image_id,
                'is_syn': is_syn,
                'flipped': False}


    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """

        image_path_jpg = os.path.join(self._data_path, index + '_color.jpg')
        image_path_png = os.path.join(self._data_path, index + '_color.png')
        if os.path.exists(image_path_jpg):
            return image_path_jpg
        elif os.path.exists(image_path_png):
            return image_path_png

        assert os.path.exists(image_path_jpg) or os.path.exists(image_path_png), \
                'Path does not exist: {} or {}'.format(image_path_jpg, image_path_png)


    def depth_path_from_index(self, index):
        """
        Construct an depth path from the image's "index" identifier.
        """
        depth_path = os.path.join(self._data_path, index + '_depth' + self._image_ext)
        assert os.path.exists(depth_path), \
                'Path does not exist: {}'.format(depth_path)
        return depth_path


    def metadata_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        metadata_path = os.path.join(self._data_path, index + '_meta.mat')
        assert os.path.exists(metadata_path), \
                'Path does not exist: {}'.format(metadata_path)
        return metadata_path
