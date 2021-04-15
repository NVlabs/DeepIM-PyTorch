# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os
import sys
import yaml
import numpy as np
import torch
import torch.utils.data as data
import numpy as np
import numpy.random as npr
import cv2
import copy
import glob
import scipy
import os.path as osp

import datasets
from fcn.config import cfg
from utils.blob import pad_im, chromatic_transform, add_noise, add_noise_depth
from transforms3d.quaternions import mat2quat, quat2mat
from utils.se3 import *
from utils.pose_error import *
from utils.cython_bbox import bbox_overlaps

_SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]

_SERIALS = [
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

_MANO_JOINTS = [
    'wrist',
    'thumb_mcp',
    'thumb_pip',
    'thumb_dip',
    'thumb_tip',
    'index_mcp',
    'index_pip',
    'index_dip',
    'index_tip',
    'middle_mcp',
    'middle_pip',
    'middle_dip',
    'middle_tip',
    'ring_mcp',
    'ring_pip',
    'ring_dip',
    'ring_tip',
    'little_mcp',
    'little_pip',
    'little_dip',
    'little_tip'
]

_MANO_JOINT_CONNECT = [
    [0,  1], [ 1,  2], [ 2,  3], [ 3,  4],
    [0,  5], [ 5,  6], [ 6,  7], [ 7,  8],
    [0,  9], [ 9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]

_BOP_EVAL_SUBSAMPLING_FACTOR = 4

class DexYCBDataset(data.Dataset, datasets.imdb):

    def __init__(self, setup, split):
        self._setup = setup
        self._split = split
        self._color_format = "color_{:06d}.jpg"
        self._depth_format = "aligned_depth_to_color_{:06d}.png"
        self._label_format = "labels_{:06d}.npz"
        self._height = 480
        self._width = 640

        # paths
        self._name = 'dex_ycb_' + setup + '_' + split
        self._image_set = split
        self._dex_ycb_path = self._get_default_path()
        path = os.path.join(self._dex_ycb_path, 'data')
        self._data_dir = path
        self._calib_dir = os.path.join(self._data_dir, "calibration")
        self._model_dir = os.path.join(self._data_dir, "models")

        self._obj_file = {
            k: os.path.join(self._model_dir, v, "textured_simple.obj")
            for k, v in _YCB_CLASSES.items()
        }

        # define all the classes
        self._classes_all = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
        self._num_classes_all = len(self._classes_all)
        self._class_colors_all = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
                              (192, 0, 0), (0, 192, 0), (0, 0, 192)]
        self._symmetry_all = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).astype(np.float32)
        self._extents_all = self._load_object_extents()
        self._posecnn_class_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21]

        # select a subset of classes
        self._classes = [self._classes_all[i] for i in cfg.TRAIN.CLASSES]
        self._classes_test = [self._classes_all[i] for i in cfg.TEST.CLASSES]
        self._num_classes = len(self._classes)
        self._class_colors = [self._class_colors_all[i] for i in cfg.TRAIN.CLASSES]
        self._symmetry = self._symmetry_all[cfg.TRAIN.CLASSES]
        self._symmetry_test = self._symmetry_all[cfg.TEST.CLASSES]
        self._extents = self._extents_all[cfg.TRAIN.CLASSES]
        self._extents_test = self._extents_all[cfg.TEST.CLASSES]
        self._pixel_mean = cfg.PIXEL_MEANS / 255.0

        # train classes
        self._points, self._points_all, self._point_blob = \
            self._load_object_points(self._classes, self._extents, self._symmetry)

        # test classes
        self._points_test, self._points_all_test, self._point_blob_test = \
            self._load_object_points(self._classes_test, self._extents_test, self._symmetry_test)

        # 3D model paths
        self.model_mesh_paths = ['{}/{}/textured_simple.obj'.format(self._model_dir, cls) for cls in self._classes_all]
        self.model_texture_paths = ['{}/{}/texture_map.png'.format(self._model_dir, cls) for cls in self._classes_all]
        self.model_colors = [np.array(self._class_colors_all[i]) / 255.0 for i in range(len(self._classes_all))]

        self.model_mesh_paths_target = ['{}/{}/textured_simple.obj'.format(self._model_dir, cls) for cls in self._classes]
        self.model_texture_paths_target = ['{}/{}/texture_map.png'.format(self._model_dir, cls) for cls in self._classes]
        self.model_colors_target = [np.array(self._class_colors_all[i]) / 255.0 for i in cfg.TRAIN.CLASSES]

        # Seen subjects, camera views, grasped objects.
        if self._setup == 's0':
            if self._split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i % 5 != 4]
            if self._split == 'val':
                subject_ind = [0, 1]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i % 5 == 4]
            if self._split == 'test':
                subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i % 5 == 4]

        # Unseen subjects.
        if self._setup == 's1':
            if self._split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = list(range(100))
            if self._split == 'val':
                subject_ind = [6]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = list(range(100))
            if self._split == 'test':
                subject_ind = [7, 8]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = list(range(100))

        # Unseen camera views.
        if self._setup == 's2':
            if self._split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5]
                sequence_ind = list(range(100))
            if self._split == 'val':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [6]
                sequence_ind = list(range(100))
            if self._split == 'test':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [7]
                sequence_ind = list(range(100))

        # Unseen grasped objects.
        if self._setup == 's3':
            if self._split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [
                    i for i in range(100) if i // 5 not in (3, 7, 11, 15, 19)
                ]
            if self._split == 'val':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i // 5 in (3, 19)]
            if self._split == 'test':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i // 5 in (7, 11, 15)]

        self._subjects = [_SUBJECTS[i] for i in subject_ind]
        self._serials = [_SERIALS[i] for i in serial_ind]
        self._intrinsics = []
        for s in self._serials:
            intr_file = os.path.join(self._calib_dir, "intrinsics", "{}_{}x{}.yml".format(s, self._width, self._height))
            with open(intr_file, 'r') as f:
                intr = yaml.load(f, Loader=yaml.FullLoader)
            intr = intr['color']
            self._intrinsics.append(intr)

        # build mapping
        self._sequences = []
        self._mapping = []
        self._ycb_ids = []
        offset = 0
        for n in self._subjects:
            seq = sorted(os.listdir(os.path.join(self._data_dir, n)))
            seq = [os.path.join(n, s) for s in seq]
            assert len(seq) == 100
            seq = [seq[i] for i in sequence_ind]
            self._sequences += seq
            for i, q in enumerate(seq):
                meta_file = os.path.join(self._data_dir, q, "meta.yml")
                with open(meta_file, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
                c = np.arange(len(self._serials))
                f = np.arange(meta['num_frames'])
                f, c = np.meshgrid(f, c)
                c = c.ravel()
                f = f.ravel()
                s = (offset + i) * np.ones_like(c)
                m = np.vstack((s, c, f)).T
                self._mapping.append(m)
                self._ycb_ids.append(meta['ycb_ids'])
            offset += len(seq)
        self._mapping = np.vstack(self._mapping)

        # sample a subset for training
        if split == 'train':
            self._mapping = self._mapping[::10]

        # dataset size
        self._size = len(self._mapping)
        print('dataset %s with images %d' % (self._name, self._size))


    def __len__(self):
        return self._size


    def get_bop_id_from_idx(self, idx):
        s, c, f = map(lambda x: x.item(), self._mapping[idx])
        scene_id = s * len(self._serials) + c
        im_id = f
        return scene_id, im_id


    def __getitem__(self, idx):
        s, c, f = self._mapping[idx]

        is_testing = f % _BOP_EVAL_SUBSAMPLING_FACTOR == 0
        if self._split == 'test' and not is_testing:
            sample = {'is_testing': is_testing}
            return sample

        scene_id, im_id = self.get_bop_id_from_idx(idx)
        video_id = '%04d' % (scene_id)
        image_id = '%06d' % (im_id)

        # posecnn result path
        posecnn_result_path = os.path.join(self._dex_ycb_path, 'results_posecnn', self._name, \
            'vgg16_dex_ycb_epoch_16.checkpoint.pth', video_id + '_' + image_id + '.mat')

        d = os.path.join(self._data_dir, self._sequences[s], self._serials[c])
        roidb = {
            'color_file': os.path.join(d, self._color_format.format(f)),
            'depth_file': os.path.join(d, self._depth_format.format(f)),
            'label_file': os.path.join(d, self._label_format.format(f)),
            'intrinsics': self._intrinsics[c],
            'ycb_ids': self._ycb_ids[s],
            'posecnn': posecnn_result_path,
        }

        # Get the input image blob
        random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
        im_blob, im_depth, im_scale, height, width = self._get_image_blob(roidb['color_file'], roidb['depth_file'], random_scale_ind)

        # build the label blob
        meta_data_blob, label_blob, mask, pose_blob, gt_boxes, poses_result, rois_result, im_depth_tensor \
            = self._get_label_blob(roidb, self._num_classes, im_scale, im_depth, height, width)

        is_syn = 0
        im_info = np.array([im_blob.shape[1], im_blob.shape[2], im_scale, is_syn], dtype=np.float32)

        sample = {'image_color': im_blob,
                  'image_depth': im_depth_tensor,
                  'meta_data': meta_data_blob,
                  'label_blob': label_blob,
                  'mask': mask,
                  'poses': pose_blob,
                  'extents': self._extents,
                  'points': self._point_blob,
                  'gt_boxes': gt_boxes,
                  'poses_result': poses_result,
                  'rois_result': rois_result,
                  'im_info': im_info,
                  'video_id': video_id,
                  'image_id': image_id}

        if self._split == 'test':
            sample['is_testing'] = is_testing

        return sample


    def _get_image_blob(self, color_file, depth_file, scale_ind):    

        # rgba
        rgba = pad_im(cv2.imread(color_file, cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]
        if im_scale != 1.0:
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        height = im.shape[0]
        width = im.shape[1]

        # chromatic transform
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = add_noise(im)
        im_tensor = torch.from_numpy(im) / 255.0
        im_tensor -= self._pixel_mean

        # depth image
        im_depth = pad_im(cv2.imread(depth_file, cv2.IMREAD_UNCHANGED), 16)
        if im_scale != 1.0:
            im_depth = cv2.resize(im_depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
        im_depth = im_depth.astype('float') / 1000.0

        return im_tensor, im_depth, im_scale, height, width


    def _get_label_blob(self, roidb, num_classes, im_scale, im_depth, height, width):
        """ build the label blob """

        # parse data
        cls_indexes = roidb['ycb_ids']
        classes = np.array(cfg.TRAIN.CLASSES)
        fx = roidb['intrinsics']['fx']
        fy = roidb['intrinsics']['fy']
        px = roidb['intrinsics']['ppx']
        py = roidb['intrinsics']['ppy']
        intrinsic_matrix = np.eye(3, dtype=np.float32)
        intrinsic_matrix[0, 0] = fx
        intrinsic_matrix[1, 1] = fy
        intrinsic_matrix[0, 2] = px
        intrinsic_matrix[1, 2] = py
        label = np.load(roidb['label_file'])

        # read label image
        im_label = label['seg']
        if im_scale != 1.0:
            im_label = cv2.resize(im_label, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)

        label_blob = np.zeros((num_classes, height, width), dtype=np.float32)
        num_pixels = np.zeros((num_classes, ), dtype=np.int32)
        for i in range(num_classes):
            I = np.where(im_label == classes[i]+1)
            num_pixels[i] = len(I[0])
            if len(I[0]) > 0:
                label_blob[i, I[0], I[1]] = 1.0

        # foreground mask
        seg = torch.from_numpy((im_label != 0).astype(np.float32))
        mask = seg.unsqueeze(2).repeat((1, 1, 3)).float()

        # poses
        poses = label['pose_y']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (1, 3, 4))
        num = poses.shape[0]
        assert num == len(cls_indexes), 'number of poses not equal to number of objects'

        # compute bounding boxes
        boxes = np.zeros((num, 4), dtype=np.float32)
        for i in range(num):
            cls = int(cls_indexes[i]) - 1
            ind = np.where(classes == cls)[0]
            if len(ind) > 0:
                x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
                x3d[0, :] = self._points_all[ind,:,0]
                x3d[1, :] = self._points_all[ind,:,1]
                x3d[2, :] = self._points_all[ind,:,2]
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = poses[i, :, :3]
                RT[:, 3] = poses[i, :, 3]
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        
                boxes[i, 0] = np.min(x2d[0, :])
                boxes[i, 1] = np.min(x2d[1, :])
                boxes[i, 2] = np.max(x2d[0, :])
                boxes[i, 3] = np.max(x2d[1, :])

        # construct pose and box blob
        pose_blob = np.zeros((num, 9), dtype=np.float32)
        gt_boxes = np.zeros((num, 5), dtype=np.float32)
        for j in range(num):
            R = poses[j, :, :3]
            T = poses[j, :, 3]

            pose_blob[j, 0] = 1
            pose_blob[j, 1] = cls_indexes[j]
            qt = mat2quat(R)
            if qt[0] < 0:
                qt = -1 * qt
            pose_blob[j, 2:6] = qt
            pose_blob[j, 6:] = T

            gt_boxes[j, :4] =  boxes[j, :] * im_scale
            gt_boxes[j, 4] =  cls_indexes[j]

        # select the classes
        index = []
        for i in range(num):
            cls = int(pose_blob[i, 1]) - 1
            ind = np.where(classes == cls)[0]
            if len(ind) > 0 and num_pixels[ind] > cfg.TRAIN.MAX_PXIEL_PER_OBJECT:
                index.append(i)
                pose_blob[i, 1] = ind
                gt_boxes[i, 4] = ind
        pose_blob = pose_blob[index, :]
        gt_boxes = gt_boxes[index, :]

        # make the same size for pose and box
        if cfg.MODE == 'TRAIN' or cfg.TEST.IMS_PER_BATCH > 1:
            pose_blob_final = np.zeros((cfg.TRAIN.MAX_OBJECT_PER_IMAGE, 9), dtype=np.float32)
            gt_boxes_final = np.zeros((cfg.TRAIN.MAX_OBJECT_PER_IMAGE, 5), dtype=np.float32)
            n = min(cfg.TRAIN.MAX_OBJECT_PER_IMAGE, pose_blob.shape[0])
            index = np.random.permutation(pose_blob.shape[0])[:n]
            pose_blob_final[:n, :] = pose_blob[index, :]
            gt_boxes_final[:n, :] = gt_boxes[index, :]
        else:
            pose_blob_final = pose_blob
            gt_boxes_final = gt_boxes

        # construct the meta data
        """
        format of the meta_data
        intrinsic matrix: meta_data[0 ~ 8]
        inverse intrinsic matrix: meta_data[9 ~ 17]
        """
        K = np.matrix(intrinsic_matrix) * im_scale
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        meta_data_blob = np.zeros(18, dtype=np.float32)
        meta_data_blob[0:9] = K.flatten()
        meta_data_blob[9:18] = Kinv.flatten()

        # compute point cloud
        fx = K[0, 0]
        fy = K[1, 1]
        px = K[0, 2]
        py = K[1, 2]
        im_xyz = self.compute_xyz(im_depth, fx, fy, px, py)
        im_depth_tensor = torch.from_numpy(im_xyz).float()
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN':
                im_depth_tensor = add_noise_depth(im_depth_tensor)

        # load posecnn result if available
        if osp.exists(roidb['posecnn']):
            result = scipy.io.loadmat(roidb['posecnn'])
            n = result['poses'].shape[0]
            poses_result = np.zeros((n, 9), dtype=np.float32)
            poses_result[:, 0] = 1
            poses_result[:, 1] = result['rois'][:, 1]
            poses_result[:, 2:] = result['poses']
            rois_result = result['rois'].copy()

            # select the classes
            index = []
            for i in range(poses_result.shape[0]):
                cls = self._posecnn_class_indexes[int(poses_result[i, 1])] - 1
                ind = np.where(classes == cls)[0]
                if len(ind) > 0:
                    index.append(i)
                    poses_result[i, 1] = ind
                    rois_result[i, 1] = ind
            poses_result = poses_result[index, :]
            rois_result = rois_result[index, :]
        else:
            print('no posecnn result %s' % (roidb['posecnn']))
            poses_result = np.zeros((1, 9), dtype=np.float32)
            rois_result = np.zeros((1, 7), dtype=np.float32)
    
        return meta_data_blob, label_blob, mask, pose_blob_final, gt_boxes_final, poses_result, rois_result, im_depth_tensor


    def _get_default_path(self):
        """
        Return the default path where YCB_Video is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'DEX_YCB')


    def _load_object_extents(self):
        extents = np.zeros((self._num_classes_all, 3), dtype=np.float32)
        for i in range(self._num_classes_all):
            point_file = os.path.join(self._model_dir, self._classes_all[i], 'points.xyz')
            print(point_file)
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points = np.loadtxt(point_file)
            extents[i, :] = 2 * np.max(np.absolute(points), axis=0)
        return extents


    def _load_object_points(self, classes, extents, symmetry):

        points = [[] for _ in range(len(classes))]
        num = np.inf

        for i in range(len(classes)):
            point_file = os.path.join(self._model_dir, classes[i], 'points.xyz')
            print(point_file)
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[i] = np.loadtxt(point_file)
            if points[i].shape[0] < num:
                num = points[i].shape[0]

        points_all = np.zeros((len(classes), num, 3), dtype=np.float32)
        for i in range(len(classes)):
            points_all[i, :, :] = points[i][:num, :]

        # rescale the points
        point_blob = points_all.copy()
        for i in range(len(classes)):
            # compute the rescaling factor for the points
            # weight = 2.0 / np.amax(extents[i, :])
            weight = 1.0
            point_blob[i, :, :] = weight * point_blob[i, :, :]

        return points, points_all, point_blob


    # compute box
    def compute_box(self, cls, intrinsic_matrix, RT):
        classes = np.array(cfg.TRAIN.CLASSES)
        ind = np.where(classes == cls)[0]
        x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
        x3d[0, :] = self._points_all[ind,:,0]
        x3d[1, :] = self._points_all[ind,:,1]
        x3d[2, :] = self._points_all[ind,:,2]
        x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        x1 = np.min(x2d[0, :])
        y1 = np.min(x2d[1, :])
        x2 = np.max(x2d[0, :])
        y2 = np.max(x2d[1, :])
        return [x1, y1, x2, y2]


    def write_dop_results(self, output_dir):
        # only write the result file
        filename = os.path.join(output_dir, 'deepim_' + self.name + '_' + cfg.INPUT + '.csv')
        f = open(filename, 'w')
        f.write('scene_id,im_id,obj_id,score,R,t,time\n')

        # list the mat file
        images_color = []
        filename = os.path.join(output_dir, '*.mat')
        files = sorted(glob.glob(filename))

        # for each image
        for i in range(len(files)):
            filename = os.path.basename(files[i])

            # parse filename
            pos = filename.find('_')
            scene_id = int(filename[:pos])
            im_id = int(filename[pos+1:-4])

            # load result
            print(files[i])
            result = scipy.io.loadmat(files[i])
            if len(result['rois']) == 0:
                continue

            rois = result['rois']
            num = rois.shape[0]
            for j in range(num):
                obj_id = cfg.TRAIN.CLASSES[int(rois[j, 1])] + 1
                score = rois[j, -1]
                run_time = -1

                # pose from network of the last iteration
                R = quat2mat(result['poses_est'][-1][j, 2:6].flatten())
                t = result['poses_est'][-1][j, 6:] * 1000
                line = '{scene_id},{im_id},{obj_id},{score},{R},{t},{time}\n'.format(
                    scene_id=scene_id,
                    im_id=im_id,
                    obj_id=obj_id,
                    score=score,
                    R=' '.join(map(str, R.flatten().tolist())),
                    t=' '.join(map(str, t.flatten().tolist())),
                    time=run_time)
                f.write(line)

        # close file
        f.close()


    def evaluation(self, output_dir):
        self.write_dop_results(output_dir)

        filename = os.path.join(output_dir, 'results_deepim.mat')
        num_iterations = cfg.TEST.ITERNUM
        if os.path.exists(filename):
            results_all = scipy.io.loadmat(filename)
            print('load results from file')
            print(filename)
            distances_sys = results_all['distances_sys']
            distances_non = results_all['distances_non']
            errors_rotation = results_all['errors_rotation']
            errors_translation = results_all['errors_translation']
            results_seq_id = results_all['results_seq_id'].flatten()
            results_frame_id = results_all['results_frame_id'].flatten()
            results_object_id = results_all['results_object_id'].flatten()
            results_cls_id = results_all['results_cls_id'].flatten()
        else:
            # save results
            num_max = 200000
            num_results = num_iterations + 1
            distances_sys = np.zeros((num_max, num_results), dtype=np.float32)
            distances_non = np.zeros((num_max, num_results), dtype=np.float32)
            errors_rotation = np.zeros((num_max, num_results), dtype=np.float32)
            errors_translation = np.zeros((num_max, num_results), dtype=np.float32)
            results_seq_id = np.zeros((num_max, ), dtype=np.float32)
            results_frame_id = np.zeros((num_max, ), dtype=np.float32)
            results_object_id = np.zeros((num_max, ), dtype=np.float32)
            results_cls_id = np.zeros((num_max, ), dtype=np.float32)

            # for each image
            count = -1
            for i in range(len(self._mapping)):

                s, c, f = self._mapping[i]
                is_testing = f % _BOP_EVAL_SUBSAMPLING_FACTOR == 0
                if not is_testing:
                    continue
    
                # intrinsics
                intrinsics = self._intrinsics[c]
                intrinsic_matrix = np.eye(3, dtype=np.float32)
                intrinsic_matrix[0, 0] = intrinsics['fx']
                intrinsic_matrix[1, 1] = intrinsics['fy']
                intrinsic_matrix[0, 2] = intrinsics['ppx']
                intrinsic_matrix[1, 2] = intrinsics['ppy']

                # parse keyframe name
                scene_id, im_id = self.get_bop_id_from_idx(i)

                # load result
                filename = os.path.join(output_dir, '%04d_%06d.mat' % (scene_id, im_id))
                print(filename)
                if not os.path.exists(filename):
                    print('file %s not exist' % (filename))
                    continue
                result = scipy.io.loadmat(filename)

                # load gt
                d = os.path.join(self._data_dir, self._sequences[s], self._serials[c])
                label_file = os.path.join(d, self._label_format.format(f))
                label = np.load(label_file)
                cls_indexes = np.array(self._ycb_ids[s]).flatten()

                # poses
                poses = label['pose_y']
                if len(poses.shape) == 2:
                    poses = np.reshape(poses, (1, 3, 4))
                num = poses.shape[0]
                assert num == len(cls_indexes), 'number of poses not equal to number of objects'

                # instance label
                im_label = label['seg']
                instance_ids = np.unique(im_label)
                if instance_ids[0] == 0:
                    instance_ids = instance_ids[1:]
                if instance_ids[-1] == 255:
                    instance_ids = instance_ids[:-1]

                # for each gt poses
                for j in range(len(instance_ids)):
                    cls = instance_ids[j]

                    # find the number of pixels of the object
                    pixels = np.sum(im_label == cls)
                    if pixels < 200:
                        continue
                    count += 1

                    # find the pose
                    object_index = np.where(cls_indexes == cls)[0][0]
                    RT_gt = poses[object_index, :, :]
                    box_gt = self.compute_box(cls - 1, intrinsic_matrix, RT_gt)

                    results_seq_id[count] = scene_id
                    results_frame_id[count] = im_id
                    results_object_id[count] = object_index
                    results_cls_id[count] = cls

                    # network result
                    roi_index = []
                    if len(result['rois']) > 0:
                        for k in range(result['rois'].shape[0]):
                            ind = int(result['rois'][k, 1])
                            if cls == cfg.TRAIN.CLASSES[ind] + 1:
                                roi_index.append(k)

                    # select the roi
                    if len(roi_index) > 1:
                        # overlaps: (rois x gt_boxes)
                        roi_blob = result['rois'][roi_index, :]
                        roi_blob = roi_blob[:, (0, 2, 3, 4, 5, 1)]
                        gt_box_blob = np.zeros((1, 5), dtype=np.float32)
                        gt_box_blob[0, 1:] = box_gt
                        overlaps = bbox_overlaps(
                            np.ascontiguousarray(roi_blob[:, :5], dtype=np.float),
                            np.ascontiguousarray(gt_box_blob, dtype=np.float)).flatten()
                        assignment = overlaps.argmax()
                        roi_index = [roi_index[assignment]]

                    if len(roi_index) > 0:
                        RT = np.zeros((3, 4), dtype=np.float32)
                        ind = int(result['rois'][roi_index, 1])
                        points = self._points[ind]

                        # initial pose
                        RT[:3, :3] = quat2mat(result['poses_init'][roi_index, 2:6].flatten())
                        RT[:, 3] = result['poses_init'][roi_index, 6:]
                        distances_sys[count, 0] = adi(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points)
                        distances_non[count, 0] = add(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points)
                        errors_rotation[count, 0] = re(RT[:3, :3], RT_gt[:3, :3])
                        errors_translation[count, 0] = te(RT[:, 3], RT_gt[:, 3])

                        # pose after refinement
                        for k in range(num_iterations):
                            RT[:3, :3] = quat2mat(result['poses_est'][k][roi_index, 2:6].flatten())
                            RT[:, 3] = result['poses_est'][k][roi_index, 6:]
                            distances_sys[count, k+1] = adi(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points)
                            distances_non[count, k+1] = add(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points)
                            errors_rotation[count, k+1] = re(RT[:3, :3], RT_gt[:3, :3])
                            errors_translation[count, k+1] = te(RT[:, 3], RT_gt[:, 3])
                    else:
                        distances_sys[count, :] = np.inf
                        distances_non[count, :] = np.inf
                        errors_rotation[count, :] = np.inf
                        errors_translation[count, :] = np.inf

            distances_sys = distances_sys[:count+1, :]
            distances_non = distances_non[:count+1, :]
            errors_rotation = errors_rotation[:count+1, :]
            errors_translation = errors_translation[:count+1, :]
            results_seq_id = results_seq_id[:count+1]
            results_frame_id = results_frame_id[:count+1]
            results_object_id = results_object_id[:count+1]
            results_cls_id = results_cls_id[:count+1]

            results_all = {'distances_sys': distances_sys,
                       'distances_non': distances_non,
                       'errors_rotation': errors_rotation,
                       'errors_translation': errors_translation,
                       'results_seq_id': results_seq_id,
                       'results_frame_id': results_frame_id,
                       'results_object_id': results_object_id,
                       'results_cls_id': results_cls_id }

            filename = os.path.join(output_dir, 'results_deepim.mat')
            scipy.io.savemat(filename, results_all)

        # print the results
        # for each class
        import matplotlib.pyplot as plt
        max_distance = 0.1
        color = ['r', 'g', 'b', 'y', 'c']
        index_plot = [0]
        leng = ['Initial']
        for k in range(num_iterations):
            leng.append('Iteration %d' % (k + 1))
            index_plot.append(k + 1)
        num = len(leng)
        ADD = np.zeros((self._num_classes_all + 1, num), dtype=np.float32)
        ADDS = np.zeros((self._num_classes_all + 1, num), dtype=np.float32)
        TS = np.zeros((self._num_classes_all + 1, num), dtype=np.float32)
        classes = list(copy.copy(self._classes_all))
        classes.append('all')
        for k in range(self._num_classes_all + 1):
            fig = plt.figure(figsize=(16.0, 14.0))
            if k == self._num_classes_all:
                index = range(len(results_cls_id))
            else:
                index = np.where(results_cls_id == k + 1)[0]

            if len(index) == 0:
                continue
            print('%s: %d objects' % (classes[k], len(index)))

            # distance symmetry
            ax = fig.add_subplot(3, 3, 1)
            lengs = []
            for i in index_plot:
                D = distances_sys[index, i]
                ind = np.where(D > max_distance)[0]
                D[ind] = np.inf
                d = np.sort(D)
                n = len(d)
                accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
                plt.plot(d, accuracy, color[i], linewidth=2)
                ADDS[k, i] = VOCap(d, accuracy)
                lengs.append('%s (%.2f)' % (leng[i], ADDS[k, i] * 100))
                print('%s, %s: %d objects missed' % (classes[k], leng[i], np.sum(np.isinf(D))))

            ax.legend(lengs)
            plt.xlabel('Average distance threshold in meter (symmetry)')
            plt.ylabel('accuracy')
            ax.set_title(classes[k])

            # distance non-symmetry
            ax = fig.add_subplot(3, 3, 2)
            lengs = []
            for i in index_plot:
                D = distances_non[index, i]
                ind = np.where(D > max_distance)[0]
                D[ind] = np.inf
                d = np.sort(D)
                n = len(d)
                accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
                plt.plot(d, accuracy, color[i], linewidth=2)
                ADD[k, i] = VOCap(d, accuracy)
                lengs.append('%s (%.2f)' % (leng[i], ADD[k, i] * 100))
                print('%s, %s: %d objects missed' % (classes[k], leng[i], np.sum(np.isinf(D))))

            ax.legend(lengs)
            plt.xlabel('Average distance threshold in meter (non-symmetry)')
            plt.ylabel('accuracy')
            ax.set_title(classes[k])

            # translation
            ax = fig.add_subplot(3, 3, 3)
            lengs = []
            for i in index_plot:
                D = errors_translation[index, i]
                ind = np.where(D > max_distance)[0]
                D[ind] = np.inf
                d = np.sort(D)
                n = len(d)
                accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
                plt.plot(d, accuracy, color[i], linewidth=2)
                TS[k, i] = VOCap(d, accuracy)
                lengs.append('%s (%.2f)' % (leng[i], TS[k, i] * 100))
                print('%s, %s: %d objects missed' % (classes[k], leng[i], np.sum(np.isinf(D))))

            ax.legend(lengs)
            plt.xlabel('Translation threshold in meter')
            plt.ylabel('accuracy')
            ax.set_title(classes[k])

            # rotation histogram
            count = 4
            for i in index_plot:
                ax = fig.add_subplot(3, 3, count)
                D = errors_rotation[index, i]
                ind = np.where(np.isfinite(D))[0]
                D = D[ind]
                ax.hist(D, bins=range(0, 190, 10), range=(0, 180))
                plt.xlabel('Rotation angle error')
                plt.ylabel('count')
                ax.set_title(leng[i])
                count += 1

            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            filename = output_dir + '/' + classes[k] + '.png'
            plt.savefig(filename)
            # plt.show()

        # print ADD
        for i in range(cfg.TEST.ITERNUM + 1):
            if i == 0:
                prefix = 'Initial'
            else:
                prefix = 'Iteration %d' % (i)
            print('==================ADD %s======================' % (prefix))
            for k in range(len(classes)):
                print('%s: %f' % (classes[k], ADD[k, i]))

            print('mean: %f' % (np.mean(ADD[:-1, i])))

            for k in range(len(classes)):
                print('%f' % (ADD[k, i]))
            print(cfg.TRAIN.SNAPSHOT_INFIX)
            print('===========================================')

            # print ADD-S
            print('==================ADD-S %s====================' % (prefix))
            for k in range(len(classes)):
                print('%s: %f' % (classes[k], ADDS[k, i]))

            print('mean: %f' % (np.mean(ADDS[:-1, i])))

            for k in range(len(classes)):
                print('%f' % (ADDS[k, i]))
            print(cfg.TRAIN.SNAPSHOT_INFIX)
            print('===========================================')
