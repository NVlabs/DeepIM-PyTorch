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
try:
    import cPickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle as cPickle
import scipy.io

import datasets
from fcn.config import cfg
from utils.blob import pad_im, chromatic_transform, add_noise, add_noise_cuda, add_noise_depth_cuda, add_noise_depth
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import mat2euler, euler2mat, euler2quat
from utils.pose_error import *
from utils.se3 import *

class YCBObject(data.Dataset, datasets.imdb):
    def __init__(self, image_set, ycb_object_path = None):

        self._name = 'ycb_object_' + image_set
        self._image_set = image_set
        self._model_path = os.path.join(datasets.ROOT_DIR, 'data', 'models')

        self._classes_all = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick', 'holiday_cup1', 'holiday_cup2', 'sanning_mug', '001_chips_can')
        self._num_classes_all = len(self._classes_all)
        self._class_colors_all = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (0, 64, 0), (64, 0, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
                              (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192),(32, 0, 0)]
        self._symmetry_all = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        self._extents_all = self._load_object_extents()

        self._intrinsic_matrix = np.array([[524.7917885754071, 0, 332.5213232846151],
                                          [0, 489.3563960810721, 281.2339855172282],
                                          [0, 0, 1]])
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
        self._pixel_mean = cfg.PIXEL_MEANS / 255.0
        self._size = cfg.TRAIN.SYNNUM
        self._build_uniform_poses()

        # for evaluation
        self._correct_poses = np.zeros((cfg.TEST.ITERNUM+1, 3), dtype=np.float32)
        self._total_poses = 0

        assert os.path.exists(self._model_path), \
                'ycb_object model path does not exist: {}'.format(self._model_path)


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
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            pc_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        else:
            pc_tensor = None
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
            if pc_tensor is not None:
                pc_tensor = pc_tensor.flip(0)
            im_label = seg_tensor.cpu().numpy()
            im_label = im_label[:, :, (2, 1, 0)] * 255
            im_label = np.round(im_label).astype(np.uint8)
            im_label = np.clip(im_label, 0, 255)
            im_label_only, im_label = self.process_label_image(im_label)

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

        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            # XYZ coordinates in camera frame
            im_depth = pc_tensor.cpu().numpy()
            im_depth = im_depth[:, :, :3]

        label_blob = np.zeros((self.num_classes, height, width), dtype=np.float32)
        for i in range(self.num_classes):
            I = np.where(im_label == classes[i]+1)
            if len(I[0]) > 0:
                label_blob[i, I[0], I[1]] = 1.0

        # foreground mask
        seg = seg_tensor[:,:,2] + 256*seg_tensor[:,:,1] + 256*256*seg_tensor[:,:,0]
        mask = (seg != 0).unsqueeze(2).repeat((1, 1, 3)).float().cpu()

        '''
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(3, 2, 1)
        plt.imshow(im[:, :, (2, 1, 0)])
        ax = fig.add_subplot(3, 2, 2)
        plt.imshow(im_label)
        print(per_occ)
        ax = fig.add_subplot(3, 2, 3)
        plt.imshow(im_depth[:, :, 0])
        ax = fig.add_subplot(3, 2, 4)
        plt.imshow(im_depth[:, :, 1])
        ax = fig.add_subplot(3, 2, 5)
        plt.imshow(im_depth[:, :, 2])
        plt.show()
        '''

        # chromatic transform
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = add_noise(im)
        im_tensor = torch.from_numpy(im) / 255.0
        im_tensor -= self._pixel_mean

        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            im_depth_tensor = torch.from_numpy(im_depth).float()
            if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
                im_depth_tensor = add_noise_depth(im_depth_tensor).float()
        else:
            im_depth_tensor = im_tensor.clone()

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
        pose_result = pose_blob.copy()

        # im is pytorch tensor in gpu
        sample = {'image_color': im_tensor,
                  'image_depth': im_depth_tensor,
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

        return self._render_item()


    def __len__(self):
        return self._size


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
        if cfg.INPUT == 'COLOR':
            poses_est = pose['poses_est_color']
        else:
            poses_est = pose['poses_est_depth']
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


    def process_label_image(self, label_image):
        """
        change label image to label index
        """
        height = label_image.shape[0]
        width = label_image.shape[1]
        labels = np.zeros((height, width), dtype=np.int32)
        labels_all = np.zeros((height, width), dtype=np.int32)

        # label image is in BGR order
        index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
        for i in range(len(self._class_colors_all)):
            color = self._class_colors_all[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            labels_all[I[0], I[1]] = i + 1

            ind = np.where(np.array(cfg.TRAIN.CLASSES) == i)[0]
            if len(ind) > 0:
                labels[I[0], I[1]] = ind + 1

        return labels, labels_all
