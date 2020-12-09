#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import _init_paths
import argparse
import os, sys
import subprocess
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import mat2euler
from fcn.config import cfg, cfg_from_file, get_output_dir
from fcn.train_test import convert_to_image, render_one_poses
import scipy.io
import cv2
import numpy as np
from utils.se3 import *
from ycb_renderer import YCBRenderer
from datasets.factory import get_dataset

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a DeepIM network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)

    args = parser.parse_args()
    return args


def add_mask(mask, image):

    im2, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height = im2.shape[0]
    width = im2.shape[1]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 3)

    x, y = np.where(img[:, :, 0] > 0)
    image[x, y, :] = [0, 255, 0]

    return np.clip(image, 0, 255).astype(np.uint8)


def add_delta(delta, image):

    font = cv2.FONT_HERSHEY_SIMPLEX
    count = 0
    for i in range(8):
        if i == 0:
            text = 'rotation:'
            color = (0, 0, 255)
        elif i == 4:
            text = 'translation:'
            color = (0, 0, 255)
        else:
            text = '{:2f}'.format(np.absolute(delta[count]))
            color = (0, 255, 0)
            count += 1
        cv2.putText(image, text, (480, 50 + 25 * i), font, 0.8, color, 2, cv2.LINE_AA)
    return image


def make_gif(working_directory, filename):
    # generate gif (need ImageMagick)
    options = '-delay 8 -loop 0 -layers optimize'
    subprocess.call('convert %s %s/*.png %s' % (options, working_directory, filename), shell=True)


if __name__ == '__main__':

    args = parse_args()

    root = '../data/YCB_Video/data/0009/'
    image_ids = [1]
    num_images = 1
    anlge = 45
    height = 480
    width = 640

    cfg.TRAIN.CLASSES = [10, 14, 15]
    cfg.MODE = 'TEST' 
    cfg.TEST.SYNTHESIZE = False
    dataset = get_dataset('ycb_video_train')

    # prepare renderer
    print('loading 3D models')
    cfg.renderer = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, render_marker=False, gpu_id=args.gpu_id)
    cfg.renderer.load_objects(dataset.model_mesh_paths_target, dataset.model_texture_paths_target, dataset.model_colors_target)
    print(dataset.model_mesh_paths_target)
    cfg.renderer.set_camera_default()

    for i in image_ids:

        # load meta data
        filename = root + '{:06d}-meta.mat'.format(i+1)
        meta_data = scipy.io.loadmat(filename)
        intrinsic_matrix = meta_data['intrinsic_matrix']

        # prepare data
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))
        num = poses.shape[2]
        channel = 9
        pose_tgt_blob = np.zeros((num, channel), dtype=np.float32)
        for j in range(num):
            class_id = int(meta_data['cls_indexes'][j])
            RT = poses[:,:,j]
            print('class_id', class_id)
            print('RT', RT)

            R = RT[:, :3]
            T = RT[:, 3]
            pose_tgt_blob[j, 1] = cfg.TRAIN.CLASSES.index(class_id - 1)
            pose_tgt_blob[j, 2:6] = mat2quat(R)
            pose_tgt_blob[j, 6:] = T

        # construct source pose
        object_id = 1
        Rz = rotation_z(-float(anlge))
        R = np.dot(Rz, quat2mat(pose_tgt_blob[object_id, 2:6]))
        pose_src_blob = pose_tgt_blob.copy()
        pose_src_blob[object_id, 2:6] = mat2quat(R)
        is_sampling = 0

        RT_src = np.zeros((3, 4), dtype=np.float32)
        RT_src[:3, :3] = R
        RT_src[:, 3] = pose_src_blob[object_id, 6:]

        # naive coordinate
        print('naive coordinate')
        for k in range(3):

            if k == 0:
                dirname = os.path.join('../data', 'cache', 'demo', 'naive_coordinate_x')
                file_gif = os.path.join(dirname, '..', 'naive_coordinate_x.gif')
            elif k == 1:
                dirname = os.path.join('../data', 'cache', 'demo', 'naive_coordinate_y')
                file_gif = os.path.join(dirname, '..', 'naive_coordinate_y.gif')
            else:
                dirname = os.path.join('../data', 'cache', 'demo', 'naive_coordinate_z')
                file_gif = os.path.join(dirname, '..', 'naive_coordinate_z.gif')

            if not os.path.exists(dirname):
                os.makedirs(dirname)

            for j in range(anlge):
                poses_naive = np.zeros((1, channel), dtype=np.float32)
                poses_naive[:, 1] = pose_src_blob[object_id, 1]

                RT = np.zeros((3, 4), dtype=np.float32)
                if k == 0:
                    RT[:3, :3] = rotation_x(float(j))
                elif k == 1:
                    RT[:3, :3] = rotation_y(float(j))
                else:
                    RT[:3, :3] = rotation_z(float(j))
                RT1 = se3_mul(RT, RT_src)
                poses_naive[0, 2:6] = mat2quat(RT1[:3, :3])
                poses_naive[0, 6:] = RT1[:, 3]

                image_naive_blob = np.zeros((1, height, width, 3), dtype=np.float32)
                render_one_poses(height, width, intrinsic_matrix, poses_naive, image_naive_blob)
                image_naive_blob = convert_to_image(image_naive_blob / 255.0)

                # compute the delta pose
                delta = np.zeros((6, ), dtype=np.float32)
                R_delta = np.dot(quat2mat(pose_src_blob[object_id, 2:6]), quat2mat(poses_naive[0, 2:6]).transpose())
                T_delta = pose_src_blob[object_id, 6:] - poses_naive[0, 6:]
                delta[:3] = mat2euler(R_delta)
                delta[3:] = T_delta

                filename = os.path.join(dirname, '{:04d}.png'.format(j))
                cv2.imwrite(filename, add_delta(delta, image_naive_blob[0]))

            make_gif(dirname, file_gif)


        # model coordinate
        print('model coordinate')
        for k in range(3):

            if k == 0:
                dirname = os.path.join('../data', 'cache', 'demo', 'model_coordinate_x')
                file_gif = os.path.join(dirname, '..', 'model_coordinate_x.gif')
            elif k == 1:
                dirname = os.path.join('../data', 'cache', 'demo', 'model_coordinate_y')
                file_gif = os.path.join(dirname, '..', 'model_coordinate_y.gif')
            else:
                dirname = os.path.join('../data', 'cache', 'demo', 'model_coordinate_z')
                file_gif = os.path.join(dirname, '..', 'model_coordinate_z.gif')

            if not os.path.exists(dirname):
                os.makedirs(dirname)

            for j in range(anlge):

                poses_model = np.zeros((1, channel), dtype=np.float32)
                poses_model[:, 1] = pose_src_blob[object_id, 1]

                if k == 0:
                    R = rotation_x(float(j))
                elif k == 1:
                    R = rotation_y(float(j))
                else:
                    R = rotation_z(float(j))

                poses_model[0, 2:6] = mat2quat(np.dot(quat2mat(pose_src_blob[object_id, 2:6]), R))
                poses_model[0, 6:] = pose_src_blob[object_id, 6:]

                image_model_blob = np.zeros((1, height, width, 3), dtype=np.float32)
                render_one_poses(height, width, intrinsic_matrix, poses_model, image_model_blob)
                image_model_blob = convert_to_image(image_model_blob / 255.0)

                # compute the delta pose
                delta = np.zeros((6, ), dtype=np.float32)
                delta[:3] = mat2euler(R)

                filename = os.path.join(dirname, '{:04d}.png'.format(j))
                cv2.imwrite(filename, add_delta(delta, image_model_blob[0]))

            make_gif(dirname, file_gif)


        # model coordinate 1
        print('model coordinate 1')
        R_new = rotation_x(90)
        # compute the new pose source to make it starts from the same video
        Rs = np.dot(quat2mat(pose_src_blob[object_id, 2:6]), R_new.transpose())
        for k in range(3):

            if k == 0:
                dirname = os.path.join('../data', 'cache', 'demo', 'model_coordinate_x_1')
                file_gif = os.path.join(dirname, '..', 'model_coordinate_x_1.gif')
            elif k == 1:
                dirname = os.path.join('../data', 'cache', 'demo', 'model_coordinate_y_1')
                file_gif = os.path.join(dirname, '..', 'model_coordinate_y_1.gif')
            else:
                dirname = os.path.join('../data', 'cache', 'demo', 'model_coordinate_z_1')
                file_gif = os.path.join(dirname, '..', 'model_coordinate_z_1.gif')

            if not os.path.exists(dirname):
                os.makedirs(dirname)

            for j in range(anlge):

                poses_model = np.zeros((1, channel), dtype=np.float32)
                poses_model[:, 1] = pose_src_blob[object_id, 1]

                if k == 0:
                    R = rotation_x(float(j))
                elif k == 1:
                    R = rotation_y(float(j))
                else:
                    R = rotation_z(float(j))

                poses_model[0, 2:6] = mat2quat(np.dot(Rs, np.dot(R, R_new)))
                poses_model[0, 6:] = pose_src_blob[object_id, 6:]

                image_model_blob = np.zeros((1, height, width, 3), dtype=np.float32)
                render_one_poses(height, width, intrinsic_matrix, poses_model, image_model_blob)
                image_model_blob = convert_to_image(image_model_blob / 255.0)

                # compute the delta pose
                delta = np.zeros((6, ), dtype=np.float32)
                delta[:3] = mat2euler(R)

                filename = os.path.join(dirname, '{:04d}.png'.format(j))
                cv2.imwrite(filename, add_delta(delta, image_model_blob[0]))

            make_gif(dirname, file_gif)

        # camera coordinate
        print('camera coordinate')
        for k in range(3):

            if k == 0:
                dirname = os.path.join('../data', 'cache', 'demo', 'camera_coordinate_x')
                file_gif = os.path.join(dirname, '..', 'camera_coordinate_x.gif')
            elif k == 1:
                dirname = os.path.join('../data', 'cache', 'demo', 'camera_coordinate_y')
                file_gif = os.path.join(dirname, '..', 'camera_coordinate_y.gif')
            else:
                dirname = os.path.join('../data', 'cache', 'demo', 'camera_coordinate_z')
                file_gif = os.path.join(dirname, '..', 'camera_coordinate_z.gif')

            if not os.path.exists(dirname):
                os.makedirs(dirname)

            for j in range(anlge):
                poses_camera = np.zeros((1, channel), dtype=np.float32)
                poses_camera[:, 1] = pose_src_blob[object_id, 1]

                if k == 0:
                    R = rotation_x(float(j))
                elif k == 1:
                    R = rotation_y(float(j))
                else:
                    R = rotation_z(float(j))

                poses_camera[0, 2:6] = mat2quat(np.dot(R, quat2mat(pose_src_blob[object_id, 2:6])))
                poses_camera[0, 6:] = pose_src_blob[object_id, 6:]

                image_camera_blob = np.zeros((1, height, width, 3), dtype=np.float32)
                render_one_poses(height, width, intrinsic_matrix, poses_camera, image_camera_blob)
                image_camera_blob = convert_to_image(image_camera_blob / 255.0)

                # compute the delta pose
                delta = np.zeros((6, ), dtype=np.float32)
                R_delta = np.dot(quat2mat(pose_src_blob[object_id, 2:6]), quat2mat(poses_camera[0, 2:6]).transpose())
                T_delta = pose_src_blob[object_id, 6:] - poses_camera[0, 6:]
                delta[:3] = mat2euler(R_delta)
                delta[3:] = T_delta

                filename = os.path.join(dirname, '{:04d}.png'.format(j))
                cv2.imwrite(filename, add_delta(delta, image_camera_blob[0]))   

            make_gif(dirname, file_gif)
