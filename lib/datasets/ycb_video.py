# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.utils.data as data

import os, math, sys
import os.path as osp
from os.path import *
import numpy as np
import numpy.random as npr
import cv2
import copy
try:
    import cPickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle as cPickle
import scipy.io
import glob

import datasets
from fcn.config import cfg
from utils.blob import pad_im, chromatic_transform, add_noise, add_noise_cuda, add_noise_depth_cuda, add_noise_depth
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import mat2euler, euler2mat, euler2quat
from utils.pose_error import *
from utils.se3 import *


class YCBVideo(data.Dataset, datasets.imdb):
    def __init__(self, image_set, ycb_video_path = None):

        self._name = 'ycb_video_' + image_set
        self._image_set = image_set
        self._ycb_video_path = self._get_default_path() if ycb_video_path is None \
                            else ycb_video_path
        self._data_path = os.path.join(self._ycb_video_path, 'data')
        path = os.path.join(self._ycb_video_path, 'data', '0000')
        if not os.path.exists(path):
            path = os.path.join(self._ycb_video_path, 'YCB_Video_Dataset/YCB_Video_Dataset/YCB_Video_Dataset/data')
            if os.path.exists(path):
                self._data_path = path
            else:
                self._data_path = os.path.join(self._ycb_video_path, 'YCB_Video_Dataset/data')

        self._model_path = os.path.join(datasets.ROOT_DIR, 'data', 'models')

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

        self._intrinsic_matrix = np.array([[1.066778e+03, 0.000000e+00, 3.129869e+02], \
                                           [0.000000e+00, 1.067487e+03, 2.413109e+02], \
                                           [0.000000e+00, 0.000000e+00, 1.000000e+00]])
        self._width = 640
        self._height = 480

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
        self._image_index = self._load_image_set_index()
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

        assert os.path.exists(self._ycb_video_path), \
                'ycb_video path does not exist: {}'.format(self._ycb_video_path)
        assert os.path.exists(self._data_path), \
                'Data path does not exist: {}'.format(self._data_path)
        assert os.path.exists(self._model_path), \
                'Model path does not exist: {}'.format(self._model_path)


    def _render_item(self):

        height = cfg.TRAIN.SYN_HEIGHT
        width = cfg.TRAIN.SYN_WIDTH
        fx = self._intrinsic_matrix[0, 0]
        fy = self._intrinsic_matrix[1, 1]
        px = self._intrinsic_matrix[0, 2]
        py = self._intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.25
        bound = 0.1
        qt = np.zeros((7, ), dtype=np.float32)
        image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        pc_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
        classes = np.array(cfg.TRAIN.CLASSES)

        # sample target object
        cls_indexes = []
        cls_target = np.random.randint(len(cfg.TRAIN.CLASSES), size=1)[0]
        cls_indexes.append(cfg.TRAIN.CLASSES[cls_target])

        # sample target pose
        poses_all = []
        cls = int(cls_indexes[0])
        if self.pose_indexes[cls] >= len(self.pose_lists[cls]):
            self.pose_indexes[cls] = 0
            self.pose_lists[cls] = np.random.permutation(np.arange(len(self.eulers)))
        roll = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][0] + 15 * np.random.randn()
        pitch = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][1] + 15 * np.random.randn()
        yaw = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][2] + 15 * np.random.randn()
        qt[3:] = euler2quat(roll * math.pi / 180.0, pitch * math.pi / 180.0, yaw * math.pi / 180.0)
        self.pose_indexes[cls] += 1

        qt[0] = np.random.uniform(-bound, bound)
        qt[1] = np.random.uniform(-bound, bound)
        qt[2] = np.random.uniform(cfg.TRAIN.SYN_TNEAR, cfg.TRAIN.SYN_TFAR)

        # render target
        poses_all.append(qt.copy())
        cfg.renderer.set_poses(poses_all)
        cfg.renderer.set_light_pos(np.random.uniform(-0.5, 0.5, 3))
        intensity = np.random.uniform(0.8, 2)
        light_color = intensity * np.random.uniform(0.9, 1.1, 3)
        cfg.renderer.set_light_color(light_color)

        cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)

        seg = torch.sum(seg_tensor[:, :, :3], dim=2)
        mask = (seg != 0).cpu().numpy()

        # sample an occluder
        cls_indexes.append(0)
        poses_all.append(np.zeros((7, ), dtype=np.float32))
        while 1:

            while 1:
                cls_occ = np.random.randint(len(self._classes_all), size=1)[0]
                if cls_occ != cls_indexes[0]:
                    cls_indexes[1] = cls_occ
                    break

            # sample poses
            cls = int(cls_indexes[1])
            if self.pose_indexes[cls] >= len(self.pose_lists[cls]):
                self.pose_indexes[cls] = 0
                self.pose_lists[cls] = np.random.permutation(np.arange(len(self.eulers)))
            roll = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][0] + 15 * np.random.randn()
            pitch = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][1] + 15 * np.random.randn()
            yaw = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][2] + 15 * np.random.randn()
            qt[3:] = euler2quat(roll * math.pi / 180.0, pitch * math.pi / 180.0, yaw * math.pi / 180.0)
            self.pose_indexes[cls] += 1

            # translation, sample an object nearby
            object_id = 0
            extent = np.mean(self._extents_all[cls, :])

            flag = np.random.randint(0, 2)
            if flag == 0:
                flag = -1
            qt[0] = poses_all[object_id][0] + flag * extent * np.random.uniform(0.3, 0.5)
            if np.absolute(qt[0]) > bound:
                qt[0] = poses_all[object_id][0] - flag * extent * np.random.uniform(0.3, 0.5)

            flag = np.random.randint(0, 2)
            if flag == 0:
                flag = -1
            qt[1] = poses_all[object_id][1] + flag * extent * np.random.uniform(0.3, 0.5)
            if np.absolute(qt[1]) > bound:
                qt[1] = poses_all[object_id][1] - flag * extent * np.random.uniform(0.3, 0.5)

            qt[2] = poses_all[object_id][2] - extent * np.random.uniform(1.0, 2.0)
            if qt[2] < cfg.TRAIN.SYN_TNEAR:
                qt[2] = poses_all[object_id][2] + extent * np.random.uniform(1.0, 2.0)

            poses_all[1] = qt
            cfg.renderer.set_poses(poses_all)

            # rendering
            cfg.renderer.set_light_pos(np.random.uniform(-0.5, 0.5, 3))
            intensity = np.random.uniform(0.8, 2)
            light_color = intensity * np.random.uniform(0.9, 1.1, 3)
            cfg.renderer.set_light_color(light_color)
            cfg.renderer.render(cls_indexes, image_tensor, seg_tensor, pc2_tensor=pc_tensor)

            seg_tensor = seg_tensor.flip(0)
            pc_tensor = pc_tensor.flip(0)
            im_label = seg_tensor.cpu().numpy()
            im_label = im_label[:, :, (2, 1, 0)] * 255
            im_label = np.round(im_label).astype(np.uint8)
            im_label = np.clip(im_label, 0, 255)
            im_label = self.process_label_image(im_label, self._class_colors_all)

            # compute occlusion percentage
            mask_target = (im_label == cls_indexes[0]+1).astype(np.int32)

            per_occ = 1.0 - np.sum(mask & mask_target) / np.sum(mask)
            if per_occ < 0.5:
                break

        # RGB to BGR order
        image_tensor = image_tensor.flip(0)
        im = image_tensor.cpu().numpy()
        im = np.clip(im, 0, 1)
        im = im[:, :, (2, 1, 0)] * 255
        im = im.astype(np.uint8)

        # depth
        im_depth = pc_tensor.cpu().numpy()
        im_depth = im_depth[:, :, :3]

        label_blob = np.zeros((self.num_classes, height, width), dtype=np.float32)
        for i in range(self.num_classes):
            I = np.where(im_label == classes[i]+1)
            if len(I[0]) > 0:
                label_blob[i, I[0], I[1]] = 1.0

        # foreground mask
        seg = seg_tensor[:,:,2] + 256*seg_tensor[:,:,1] + 256*256*seg_tensor[:,:,0]
        mask = (seg != 0).unsqueeze(2).repeat((1, 1, 3)).float().cuda()

        '''
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(im[:, :, (2, 1, 0)])
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(im_label)
        plt.show()
        '''

        # chromatic transform
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)

        im_cuda = torch.from_numpy(im).cuda().float() / 255.0
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im_cuda = add_noise_cuda(im_cuda)
        im_cuda -= self._pixel_mean

        im_cuda_depth = torch.from_numpy(im_depth).cuda().float()
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN':
                im_cuda_depth = add_noise_depth_cuda(im_cuda_depth)

        # poses and boxes only for the target object
        pose_blob = np.zeros((1, 9), dtype=np.float32)
        gt_boxes = np.zeros((1, 5), dtype=np.float32)

        pose_blob[0, 0] = 1
        pose_blob[0, 1] = cls_target
        pose_blob[0, 2:6] = poses_all[0][3:]
        pose_blob[0, 6:] = poses_all[0][:3]

        # compute box
        x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
        x3d[0, :] = self._points_all[cls_target,:,0]
        x3d[1, :] = self._points_all[cls_target,:,1]
        x3d[2, :] = self._points_all[cls_target,:,2]
        RT = np.zeros((3, 4), dtype=np.float32)
        RT[:3, :3] = quat2mat(pose_blob[0, 2:6])
        RT[:, 3] = pose_blob[0, 6:]
        x2d = np.matmul(self._intrinsic_matrix, np.matmul(RT, x3d))
        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        
        gt_boxes[0, 0] = np.min(x2d[0, :])
        gt_boxes[0, 1] = np.min(x2d[1, :])
        gt_boxes[0, 2] = np.max(x2d[0, :])
        gt_boxes[0, 3] = np.max(x2d[1, :])
        gt_boxes[0, 4] = cls_target

        # construct the meta data
        K = self._intrinsic_matrix
        Kinv = np.linalg.pinv(K)
        meta_data_blob = np.zeros(18, dtype=np.float32)
        meta_data_blob[0:9] = K.flatten()
        meta_data_blob[9:18] = Kinv.flatten()

        is_syn = 1
        im_info = np.array([im.shape[0], im.shape[1], cfg.TRAIN.SCALES_BASE[0], is_syn], dtype=np.float32)
        pose_result = np.zeros((1, 9), dtype=np.float32)

        # im is pytorch tensor in gpu
        sample = {'image_color': im_cuda,
                  'image_depth': im_cuda_depth,
                  'meta_data': meta_data_blob,
                  'label_blob': label_blob,
                  'mask': mask,
                  'poses': pose_blob,
                  'extents': self._extents,
                  'points': self._point_blob,
                  'gt_boxes': gt_boxes,
                  'poses_result': pose_result,
                  'im_info': im_info}

        return sample


    def __getitem__(self, index):

        is_syn = 0
        if cfg.MODE == 'TRAIN' and cfg.TRAIN.SYNTHESIZE and index % (cfg.TRAIN.SYN_RATIO+1) != 0:
            is_syn = 1

        if is_syn:
            return self._render_item()

        if cfg.MODE == 'TRAIN' and cfg.TRAIN.SYNTHESIZE:
            index = int((index % self._size) / (cfg.TRAIN.SYN_RATIO+1))
        else:
            index = int(index % self._size) 
        roidb = self._roidb[index]

        # Get the input image blob
        random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
        im_blob, im_depth, im_scale, height, width = self._get_image_blob(roidb, random_scale_ind)

        # build the label blob
        meta_data_blob, label_blob, mask, pose_blob, gt_boxes, poses_result, rois_result, im_depth_tensor \
            = self._get_label_blob(roidb, self._num_classes, im_scale, im_depth, height, width)

        is_syn = roidb['is_syn']
        im_info = np.array([height, width, im_scale, is_syn], dtype=np.float32)

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
                  'video_id': roidb['video_id'],
                  'image_id': roidb['image_id']}

        return sample


    def _get_image_blob(self, roidb, scale_ind):    

        # rgba
        rgba = pad_im(cv2.imread(roidb['image'], cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        height = im.shape[0]
        width = im.shape[1]

        if roidb['flipped']:
            im = im[:, ::-1, :]

        # chromatic transform
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = add_noise(im)
        im_tensor = torch.from_numpy(im) / 255.0
        im_tensor -= self._pixel_mean

        # depth image
        im_depth = pad_im(cv2.imread(roidb['depth'], cv2.IMREAD_UNCHANGED), 16)
        if im_scale != 1.0:
            im_depth = cv2.resize(im_depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
        im_depth = im_depth.astype('float') / 10000.0

        return im_tensor, im_depth, im_scale, height, width


    def _get_label_blob(self, roidb, num_classes, im_scale, im_depth, blob_height, blob_width):
        """ build the label blob """

        meta_data = scipy.io.loadmat(roidb['meta_data'])
        meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()
        classes = np.array(cfg.TRAIN.CLASSES)

        # read label image
        im_label = pad_im(cv2.imread(roidb['label'], cv2.IMREAD_UNCHANGED), 16)
        im_label = cv2.resize(im_label, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
        label_blob = np.zeros((num_classes, blob_height, blob_width), dtype=np.float32)
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
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))
        if roidb['flipped']:
            poses = _flip_poses(poses, meta_data['intrinsic_matrix'], width)

        # compute bounding boxes
        num = poses.shape[2]
        boxes = np.zeros((num, 4), dtype=np.float32)
        for i in range(num):
            cls = int(meta_data['cls_indexes'][i]) - 1
            ind = np.where(classes == cls)[0]
            if len(ind) > 0:
                x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
                x3d[0, :] = self._points_all[ind,:,0]
                x3d[1, :] = self._points_all[ind,:,1]
                x3d[2, :] = self._points_all[ind,:,2]
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = poses[:, :3, i]
                RT[:, 3] = poses[:, 3, i]
                x2d = np.matmul(meta_data['intrinsic_matrix'], np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        
                boxes[i, 0] = np.min(x2d[0, :])
                boxes[i, 1] = np.min(x2d[1, :])
                boxes[i, 2] = np.max(x2d[0, :])
                boxes[i, 3] = np.max(x2d[1, :])

        # construct pose and box blob
        num = poses.shape[2]
        pose_blob = np.zeros((num, 9), dtype=np.float32)
        gt_boxes = np.zeros((num, 5), dtype=np.float32)
        for j in range(num):
            R = poses[:, :3, j]
            T = poses[:, 3, j]

            pose_blob[j, 0] = 1
            pose_blob[j, 1] = meta_data['cls_indexes'][j]
            qt = mat2quat(R)
            if qt[0] < 0:
                qt = -1 * qt
            pose_blob[j, 2:6] = qt
            pose_blob[j, 6:] = T

            gt_boxes[j, :4] =  boxes[j, :] * im_scale
            gt_boxes[j, 4] =  meta_data['cls_indexes'][j]

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
        K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
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
                cls = int(poses_result[i, 1]) - 1
                ind = np.where(classes == cls)[0]
                if len(ind) > 0:
                    index.append(i)
                    poses_result[i, 1] = ind
                    rois_result[i, 1] = ind
            poses_result = poses_result[index, :]
            rois_result = rois_result[index, :]
        else:
            poses_result = np.zeros((1, 9), dtype=np.float32)
            rois_result = np.zeros((1, 7), dtype=np.float32)
    
        return meta_data_blob, label_blob, mask, pose_blob_final, gt_boxes_final, poses_result, rois_result, im_depth_tensor


    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where YCB_Video is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'YCB_Video')


    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """

        # load cache file
        prefix = '_class'
        for i in range(len(cfg.TRAIN.CLASSES)):
            prefix += '_%d' % cfg.TRAIN.CLASSES[i]
        cache_file = os.path.join(self.cache_path, self.name + prefix + '_image_index.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                image_index = cPickle.load(fid)
            print('{} image index loaded from {}'.format(self.name, cache_file))
            print('{} training images'.format(len(image_index)))
            return image_index

        image_set_file = os.path.join(self._ycb_video_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        classes = np.array(cfg.TRAIN.CLASSES)
        image_index = []
        video_ids = set()
        video_ids_selected = set()

        count = 0
        with open(image_set_file) as f:
            for x in f.readlines():

                if count % 5 != 0 and self._image_set == 'train':
                    count += 1
                    continue
                count += 1

                index = x.rstrip('\n')
                pos = index.find('/')
                video_id = index[:pos]

                if video_id in video_ids_selected:
                    image_index.append(index)
                    continue

                if video_id not in video_ids:
                    video_ids.add(video_id)

                    # load the meta data
                    metadata_path = os.path.join(self._data_path, index + '-meta.mat')
                    meta_data = scipy.io.loadmat(metadata_path)
                    cls_indexes = meta_data['cls_indexes'].flatten()

                    for i in range(len(cls_indexes)):
                        cls = int(cls_indexes[i]) - 1
                        ind = np.where(classes == cls)[0]
                        if len(ind) > 0:
                            image_index.append(index)
                            video_ids_selected.add(video_id)
                            break

        print('{} real training images'.format(len(image_index)))

        # add synthetic data for training
        if self._image_set == 'train':
            filename = os.path.join(self._data_path + '_syn', '*.mat')
            files = glob.glob(filename)
            print('adding synthetic %d data' % (len(files)))
            for i in range(len(files)):
                # load the meta data
                meta_data = scipy.io.loadmat(files[i])
                cls_indexes = meta_data['cls_indexes'].flatten()
                if i % 20 == 0:
                    print(files[i])
                for j in range(len(cls_indexes)):
                    cls = int(cls_indexes[j]) - 1
                    ind = np.where(classes == cls)[0]
                    if len(ind) > 0:
                        filename = files[i].replace(self._data_path, '../data')[:-9]
                        image_index.append(filename)
                        break

        # save poses
        with open(cache_file, 'wb') as fid:
            cPickle.dump(image_index, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote image index to {}'.format(cache_file))

        return image_index


    def _load_object_points(self):

        points = [[] for _ in range(len(self._classes))]
        num = np.inf

        for i in range(len(self._classes)):
            point_file = os.path.join(self._model_path, self._classes[i], 'points.xyz')
            print(point_file)
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[i] = np.loadtxt(point_file)
            if points[i].shape[0] < num:
                num = points[i].shape[0]

        points_all = np.zeros((self._num_classes, num, 3), dtype=np.float32)
        for i in range(len(self._classes)):
            points_all[i, :, :] = points[i][:num, :]

        # rescale the points
        point_blob = points_all.copy()
        for i in range(self._num_classes):
            # compute the rescaling factor for the points
            # weight = 2.0 / np.amax(self._extents[i, :])
            weight = 1.0
            point_blob[i, :, :] = weight * point_blob[i, :, :]

        return points, points_all, point_blob


    def _load_object_extents(self):

        extents = np.zeros((self._num_classes_all, 3), dtype=np.float32)
        for i in range(self._num_classes_all):
            point_file = os.path.join(self._model_path, self._classes_all[i], 'points.xyz')
            print(point_file)
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points = np.loadtxt(point_file)
            extents[i, :] = 2 * np.max(np.absolute(points), axis=0)

        return extents


    def _load_all_poses(self, roidb):

        # load cache file
        prefix = '_class'
        for i in range(len(cfg.TRAIN.CLASSES)):
            prefix += '_%d' % cfg.TRAIN.CLASSES[i]
        cache_file = os.path.join(self.cache_path, self.name + prefix + '_poses.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                poses = cPickle.load(fid)
            print('{} poses loaded from {}'.format(self.name, cache_file))
            return poses

        poses = [np.zeros((0, 6), dtype=np.float32) for i in range(len(cfg.TRAIN.CLASSES))]
        classes = np.array(cfg.TRAIN.CLASSES)

        for i in range(len(roidb)):
            meta_data = scipy.io.loadmat(roidb[i]['meta_data'])
            cls_indexes = meta_data['cls_indexes'].flatten()
            gt = meta_data['poses']
            if len(gt.shape) == 2:
                gt = np.reshape(gt, (3, 4, 1))

            for j in range(len(cls_indexes)):
                cls = int(cls_indexes[j]) - 1
                ind = np.where(classes == cls)[0]
                if len(ind) > 0:
                    R = gt[:, :3, j]
                    T = gt[:, 3, j]
                    pose = np.zeros((1, 6), dtype=np.float32)
                    pose[0, :3] = mat2euler(R)
                    pose[0, 3:] = T
                    poses[int(ind)] = np.concatenate((poses[int(ind)], pose), axis=0)

        # save poses
        with open(cache_file, 'wb') as fid:
            cPickle.dump(poses, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote poses to {}'.format(cache_file))
        return poses


    # image
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])


    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """

        image_path = os.path.join(self._data_path, index + '-color.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(self._data_path, index + '-color.png')

        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path


    # depth
    def depth_path_at(self, i):
        """
        Return the absolute path to depth i in the image sequence.
        """
        return self.depth_path_from_index(self.image_index[i])

    def depth_path_from_index(self, index):
        """
        Construct an depth path from the image's "index" identifier.
        """
        depth_path = os.path.join(self._data_path, index + '-depth' + self._image_ext)
        assert os.path.exists(depth_path), \
                'Path does not exist: {}'.format(depth_path)
        return depth_path

    # label
    def label_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.label_path_from_index(self.image_index[i])

    def label_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        label_path = os.path.join(self._data_path, index + '-label' + self._image_ext)
        assert os.path.exists(label_path), \
                'Path does not exist: {}'.format(label_path)
        return label_path

    # camera pose
    def metadata_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.metadata_path_from_index(self.image_index[i])

    def metadata_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        metadata_path = os.path.join(self._data_path, index + '-meta.mat')
        assert os.path.exists(metadata_path), \
                'Path does not exist: {}'.format(metadata_path)
        return metadata_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        prefix = '_class'
        for i in range(len(cfg.TRAIN.CLASSES)):
            prefix += '_%d' % cfg.TRAIN.CLASSES[i]
        cache_file = os.path.join(self.cache_path, self.name + prefix + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_ycb_video_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def _load_ycb_video_annotation(self, index):
        """
        Load class name and meta data
        """
        # image path
        image_path = self.image_path_from_index(index)

        # depth path
        depth_path = self.depth_path_from_index(index)

        # label path
        label_path = self.label_path_from_index(index)

        # metadata path
        metadata_path = self.metadata_path_from_index(index)

        # is synthetic image or not
        if 'data_syn' in image_path:
            is_syn = 1
            video_id = ''
            image_id = ''
        else:
            is_syn = 0
            # parse image name
            pos = index.find('/')
            video_id = index[:pos]
            image_id = index[pos+1:]

        # posecnn result path
        posecnn_result_path = os.path.join(self._ycb_video_path, 'results_posecnn', index + '.mat')
        
        return {'image': image_path,
                'depth': depth_path,
                'label': label_path,
                'meta_data': metadata_path,
                'posecnn': posecnn_result_path,
                'video_id': video_id,
                'image_id': image_id,
                'is_syn': is_syn,
                'flipped': False}


    def _evaluate(self, ind, pose_est, pose_tgt, intrinsic_matrix):

        if cfg.TEST.VISUALIZE:
            print('iteration %d' % (ind))

        # for each object
        for i in range(pose_est.shape[0]):
            cls = int(pose_est[i, 1])

            RT_est = np.zeros((3, 4), dtype=np.float32)
            RT_est[:3, :3] = quat2mat(pose_est[i, 2:6])
            RT_est[:, 3] = pose_est[i, 6:]

            RT_tgt = np.zeros((3, 4), dtype=np.float32)
            RT_tgt[:3, :3] = quat2mat(pose_tgt[i, 2:6])
            RT_tgt[:, 3] = pose_tgt[i, 6:]

            # rotation and translation error
            error_rot = re(RT_est[:3, :3], RT_tgt[:3, :3])
            error_tran = te(RT_est[:, 3], RT_tgt[:, 3])

            if self._classes[cls] == 'eggbox' and error_rot > 90:
                RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                RT_est = se3_mul(RT_est, RT_z)
                error_rot = re(RT_est[:3, :3], RT_tgt[:3, :3])
                error_tran = te(RT_est[:, 3], RT_tgt[:, 3])

            if error_rot < 5.0 and error_tran < 0.05:
                self._correct_poses[ind, 0] += 1
            if cfg.TEST.VISUALIZE:
                print('obj %d, rot %.2f, tran %.4f' % (i+1, error_rot, error_tran))

            # compute 6D pose error
            if self._classes[cls] == 'eggbox' or self._classes[cls] == 'glue':
                error = adi(RT_est[:3, :3], RT_est[:, 3], RT_tgt[:3, :3], RT_tgt[:, 3], self._points[cls])
            else:
                error = add(RT_est[:3, :3], RT_est[:, 3], RT_tgt[:3, :3], RT_tgt[:, 3], self._points[cls])

            threshold = 0.1 * np.linalg.norm(self._extents[cls, :])
            if error < threshold:
                self._correct_poses[ind, 1] += 1
            if cfg.TEST.VISUALIZE:
                print('average distance error: {}'.format(error))

            # reprojection error
            error_reprojection = reproj(intrinsic_matrix, RT_est[:3, :3], RT_est[:, 3], RT_tgt[:3, :3], RT_tgt[:, 3], self._points[cls])
            if error_reprojection < 5.0:
                self._correct_poses[ind, 2] += 1
            if cfg.TEST.VISUALIZE:
                print('reprojection error: {}'.format(error_reprojection))


    def evaluate_pose(self, pose):
        """
        evaluate pose estimation
        """
        poses_init = pose['poses_init']
        poses_est = pose['poses_est']
        poses_tgt = pose['poses_tgt']
        intrinsic_matrix = pose['intrinsic_matrix']

        self._total_poses += poses_tgt.shape[0]
        self._evaluate(0, poses_init, poses_tgt, intrinsic_matrix)

        num_iterations = len(poses_est)
        for i in range(num_iterations):
            self._evaluate(i+1, poses_est[i], poses_tgt, intrinsic_matrix)


    def print_pose_accuracy(self):

        print('\n============================================')
        print('%d objects in total\n' % (self._total_poses))
        for i in range(cfg.TEST.ITERNUM + 1):
            if i == 0:
                prefix = 'initial pose'
            else:
                prefix = 'iteration %d' % (i)
            print(prefix)
            print('5 degree, 5cm accuracy: %.4f' % (self._correct_poses[i, 0] / self._total_poses))
            print('6D pose accuracy: %.4f' % (self._correct_poses[i, 1] / self._total_poses))
            print('Reprojection 2D accuracy: %.4f\n' % (self._correct_poses[i, 2] / self._total_poses))
        print('============================================')


    def process_label_image(self, label_image, class_colors):
        """
        change label image to label index
        """
        height = label_image.shape[0]
        width = label_image.shape[1]
        labels = np.zeros((height, width), dtype=np.int32)

        # label image is in BGR order
        index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
        for i in range(len(class_colors)):
            color = class_colors[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            labels[I[0], I[1]] = i + 1
        return labels


    def evaluation(self, output_dir):

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
            num_max = 100000
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
            for i in range(len(self._roidb)):
    
                # parse keyframe name
                seq_id = int(self._roidb[i]['video_id'])
                frame_id = int(self._roidb[i]['image_id'])

                # load result
                filename = os.path.join(output_dir, '%04d_%06d.mat' % (seq_id, frame_id))
                print(filename)
                result_deepim = scipy.io.loadmat(filename)

                # load gt poses
                filename = osp.join(self._data_path, '%04d/%06d-meta.mat' % (seq_id, frame_id))
                print(filename)
                gt = scipy.io.loadmat(filename)

                # for each gt poses
                cls_indexes = gt['cls_indexes'].flatten()
                for j in range(len(cls_indexes)):
                    count += 1
                    cls_index = cls_indexes[j]
                    RT_gt = gt['poses'][:, :, j]

                    results_seq_id[count] = seq_id
                    results_frame_id[count] = frame_id
                    results_object_id[count] = j
                    results_cls_id[count] = cls_index

                    # network result
                    result = result_deepim
                    roi_index = []
                    if len(result['rois']) > 0:     
                        for k in range(result['rois'].shape[0]):
                            ind = int(result['rois'][k, 1])
                            cls = cfg.TRAIN.CLASSES[ind] + 1
                            if cls == cls_index:
                                roi_index.append(k)                   

                    # select the roi
                    if len(roi_index) > 1:
                        # overlaps: (rois x gt_boxes)
                        roi_blob = result['rois'][roi_index, :]
                        roi_blob = roi_blob[:, (0, 2, 3, 4, 5, 1)]
                        gt_box_blob = np.zeros((1, 5), dtype=np.float32)
                        gt_box_blob[0, 1:] = gt['box'][j, :]
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
