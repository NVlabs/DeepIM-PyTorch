# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import time
import os
import sys
import math
import numpy as np
import cv2
import scipy
import threading
import cupy

from fcn.config import cfg
from fcn.multiscaleloss import multiscaleEPE, realEPE
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from transforms3d.euler import mat2euler, euler2mat, euler2quat, quat2euler
from utils.show_flows import *
from utils.se3 import T_inv_transform, se3_mul, se3_inverse
from utils.zoom_in import zoom_images
from utils.pose_error import re, te

class Stream:
    ptr = torch.cuda.current_stream().cuda_stream

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def compute_delta_poses(pose_src_blob, pose_tgt_blob, zoom_factor):

    num = pose_src_blob.shape[0]
    pose_delta_blob = pose_src_blob.copy()

    for i in range(num):
        R_src = quat2mat(pose_src_blob[i, 2:6])
        T_src = pose_src_blob[i, 6:]

        R_tgt = quat2mat(pose_tgt_blob[i, 2:6])
        T_tgt = pose_tgt_blob[i, 6:]

        R_delta = np.dot(R_tgt, R_src.transpose())
        T_delta = T_inv_transform(T_src, T_tgt)

        pose_delta_blob[i, 2:6] = mat2quat(R_delta)
        pose_delta_blob[i, 6] = T_delta[0] / zoom_factor[i, 0]
        pose_delta_blob[i, 7] = T_delta[1] / zoom_factor[i, 1]
        pose_delta_blob[i, 8] = T_delta[2]

    return pose_delta_blob


if sys.version_info[0] < 3:
    @cupy.util.memoize(for_each_device=True)
    def cunnex(strFunction):
        return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)
else:
    @cupy._util.memoize(for_each_device=True)
    def cunnex(strFunction):
        return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)


compute_flow = '''
extern "C" __global__ void compute_flow(float* pc_tgt, float* pc_src, float* flow_map, float* RT, 
                             float fx, float fy, float px, float py, int width, int height)
{
  const int y = threadIdx.x + blockDim.x * blockIdx.x;
  const int x = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < width && y < height) 
  {
    flow_map[(y * width + x) * 2] = 0;
    flow_map[(y * width + x) * 2 + 1] = 0;
    float X = pc_src[(y * width + x) * 3];
    float Y = pc_src[(y * width + x) * 3 + 1];
    float Z = pc_src[(y * width + x) * 3 + 2];
    if (Z > 0)
    {
      float vx = RT[0] * X + RT[1] * Y + RT[2] * Z + RT[3];
      float vy = RT[4] * X + RT[5] * Y + RT[6] * Z + RT[7];
      float vz = RT[8] * X + RT[9] * Y + RT[10] * Z + RT[11];

      // projection
      float w_proj = fx * (vx / vz) + px;
      float h_proj = fy * (vy / vz) + py;
      float z_proj = vz;
      int w_proj_i = roundf(w_proj);
      int h_proj_i = roundf(h_proj);

      if (w_proj_i >= 0 && w_proj_i < width && h_proj_i >= 0 && h_proj_i < height)
      {
        float z_tgt = pc_tgt[(h_proj_i * width + w_proj_i) * 3 + 2];
        if (fabs(z_proj - z_tgt) < 3E-3) 
        {
          flow_map[(y * width + x) * 2] = w_proj - x;
          flow_map[(y * width + x) * 2 + 1] = h_proj - y;
        }
      }
    }  
  }
}
'''

def render_poses(intrinsic_matrix, label_blob, pose_tgt_blob, pose_src_blob, points_all, train_data):

    height = cfg.TRAIN.SYN_HEIGHT
    width = cfg.TRAIN.SYN_WIDTH
    ratio = float(height) / float(width)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 6.0
    znear = 0.01
    num = pose_tgt_blob.shape[0]
    pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).cuda().float()

    image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    pcloud_tensor = torch.cuda.FloatTensor(height, width, 4).detach()

    threadsperblock = (32, 32, 1)
    blockspergrid_x = math.ceil(height / threadsperblock[0])
    blockspergrid_y = math.ceil(width / threadsperblock[1])
    blockspergrid = (int(blockspergrid_x), int(blockspergrid_y), 1)

    qt = np.zeros((7, ), dtype=np.float32)
    RT_tgt = np.zeros((3, 4), dtype=np.float32)
    RT_src = np.zeros((3, 4), dtype=np.float32)

    # set renderer
    cfg.renderer.set_light_pos([0, 0, 0])
    cfg.renderer.set_light_color([2, 2, 2])
    cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
    for i in range(num):
        image_id = int(pose_tgt_blob[i, 0])
        if cfg.MODE == 'TEST' and cfg.TEST.SYNTHESIZE == False:
            cls_index = int(pose_tgt_blob[i, 1])
        else:
            cls_index = cfg.TRAIN.CLASSES[int(pose_tgt_blob[i, 1])]

        # render target pose
        qt[:3] = pose_tgt_blob[i, 6:]
        qt[3:] = pose_tgt_blob[i, 2:6]
        cfg.renderer.set_poses([qt])
        cfg.renderer.render([cls_index], image_tensor, seg_tensor, pc2_tensor=pcloud_tensor)
        image_tensor = image_tensor.flip(0)
        pcloud_tensor = pcloud_tensor.flip(0)
        # mask the target point cloud
        mask = label_blob[image_id, int(pose_tgt_blob[i, 1]), :, :]
        mask_tensor = torch.from_numpy(np.tile(mask[:,:,np.newaxis], (1, 1, 3))).contiguous().cuda()

        im_tgt = image_tensor[:, :, (2, 1, 0)] - pixel_mean
        train_data['image_tgt_blob_color'][i].copy_(im_tgt.permute(2, 0, 1))
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            train_data['image_tgt_blob_depth'][i].copy_(pcloud_tensor[:, :, :3].permute(2, 0, 1))
        train_data['pcloud_tgt_cuda'].copy_(torch.mul(pcloud_tensor[:, :, :3], mask_tensor))

        # 3D points and target RT
        x3d = np.ones((4, points_all.shape[1]), dtype=np.float32)
        x3d[0, :] = points_all[int(pose_tgt_blob[i, 1]),:,0]
        x3d[1, :] = points_all[int(pose_tgt_blob[i, 1]),:,1]
        x3d[2, :] = points_all[int(pose_tgt_blob[i, 1]),:,2]
        RT_tgt[:3, :3] = quat2mat(pose_tgt_blob[i, 2:6])
        RT_tgt[:, 3]= pose_tgt_blob[i, 6:]

        # render source pose
        qt[:3] = pose_src_blob[i, 6:]
        qt[3:] = pose_src_blob[i, 2:6]
        cfg.renderer.set_poses([qt])
        cfg.renderer.render([cls_index], image_tensor, seg_tensor, pc2_tensor=pcloud_tensor)
        image_tensor = image_tensor.flip(0)
        pcloud_tensor = pcloud_tensor.flip(0)        
        im_src = image_tensor[:, :, (2, 1, 0)] - pixel_mean
        train_data['image_src_blob_color'][i].copy_(im_src.permute(2, 0, 1))
        train_data['pcloud_src_cuda'].copy_(pcloud_tensor[:, :, :3])
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            train_data['image_src_blob_depth'][i].copy_(pcloud_tensor[:, :, :3].permute(2, 0, 1))

        # compute box
        RT_src[:3, :3] = quat2mat(pose_src_blob[i, 2:6])
        RT_src[:, 3]= pose_src_blob[i, 6:]
        x2d = np.matmul(intrinsic_matrix, np.matmul(RT_src, x3d))
        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        obj_imgn_start_x = np.min(x2d[0, :])
        obj_imgn_start_y = np.min(x2d[1, :])
        obj_imgn_end_x = np.max(x2d[0, :])
        obj_imgn_end_y = np.max(x2d[1, :])
        obj_imgn_c = np.dot(intrinsic_matrix, pose_src_blob[i, 6:])
        zoom_c_x = obj_imgn_c[0] / obj_imgn_c[2]
        zoom_c_y = obj_imgn_c[1] / obj_imgn_c[2]

        x1 = max(int(obj_imgn_start_x), 0) 
        y1 = max(int(obj_imgn_start_y), 0)
        x2 = min(int(obj_imgn_end_x), width-1)
        y2 = min(int(obj_imgn_end_y), height-1)

        # mask region
        left_dist = zoom_c_x - obj_imgn_start_x
        right_dist = obj_imgn_end_x - zoom_c_x
        up_dist = zoom_c_y - obj_imgn_start_y
        down_dist = obj_imgn_end_y - zoom_c_y
        crop_height = np.max([ratio * right_dist, ratio * left_dist, up_dist, down_dist]) * 2 * 1.4
        crop_width = crop_height / ratio

        # affine transformation for PyTorch
        x1 = (zoom_c_x - crop_width / 2) * 2 / width - 1;
        x2 = (zoom_c_x + crop_width / 2) * 2 / width - 1;
        y1 = (zoom_c_y - crop_height / 2) * 2 / height - 1;
        y2 = (zoom_c_y + crop_height / 2) * 2 / height - 1;

        pts1 = np.float32([[x1, y1], [x1, y2], [x2, y1]])
        pts2 = np.float32([[-1, -1], [-1, 1], [1, -1]])
        affine_matrix = torch.tensor(cv2.getAffineTransform(pts2, pts1))
        train_data['affine_matrices'][i].copy_(affine_matrix)
        train_data['zoom_factor'][i, 0] = affine_matrix[0, 0]
        train_data['zoom_factor'][i, 1] = affine_matrix[1, 1]
        train_data['zoom_factor'][i, 2] = affine_matrix[0, 2]
        train_data['zoom_factor'][i, 3] = affine_matrix[1, 2]

        # compute optical flow on color
        pose = torch.tensor(se3_mul(RT_tgt, se3_inverse(RT_src))).cuda().float()
        with torch.cuda.device_of(train_data['pcloud_tgt_cuda']):
           cunnex('compute_flow')(
               grid=blockspergrid,
               block=threadsperblock,
               args=[train_data['pcloud_tgt_cuda'].data_ptr(),
                   train_data['pcloud_src_cuda'].data_ptr(),
                   train_data['flow_map_cuda'].data_ptr(),
                   pose.data_ptr(),
                   fx, fy, px, py, width, height],
               stream=Stream)
        train_data['flow_blob'][i].copy_(train_data['flow_map_cuda'].permute(2, 0, 1))


def render_one_poses(height, width, intrinsic_matrix, pose_blob, image_blob):
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 6.0
    znear = 0.01
    num = pose_blob.shape[0]
    qt = np.zeros((7, ), dtype=np.float32)

    image_tensor = torch.cuda.FloatTensor(height, width, 4)
    seg_tensor = torch.cuda.FloatTensor(height, width, 4)

    # set renderer
    cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

    # render images
    for i in range(num):
        if cfg.MODE == 'TEST' and cfg.TEST.SYNTHESIZE == False:
            cls_index = int(pose_blob[i, 1])
        else:
            cls_index = cfg.TRAIN.CLASSES[int(pose_blob[i, 1])]

        # render target pose
        qt[:3] = pose_blob[i, 6:]
        qt[3:] = pose_blob[i, 2:6]
        cfg.renderer.set_poses([qt])

        # rendering
        frame = cfg.renderer.render([cls_index], image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)

        # RGB to BGR order
        im = image_tensor.cpu().numpy()
        im = np.clip(im, 0, 1)
        im = im[:, :, (2, 1, 0)] * 255
        image_blob[i] = im


def render_image(dataset, im, poses):

    intrinsic_matrix = dataset._intrinsic_matrix
    height = im.shape[0]
    width = im.shape[1]

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 10.0
    znear = 0.01
    num = poses.shape[0]

    image_tensor = torch.cuda.FloatTensor(height, width, 4)
    seg_tensor = torch.cuda.FloatTensor(height, width, 4)
    pcloud_tensor = torch.cuda.FloatTensor(height, width, 4).detach()

    # set renderer
    cfg.renderer.set_light_pos([0, 0, 0])
    cfg.renderer.set_light_color([2, 2, 2])
    cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

    # render images
    cls_indexes = []
    poses_all = []
    for i in range(num):

        if cfg.MODE == 'TEST' and cfg.TEST.SYNTHESIZE == False:
            cls_index = int(poses[i, 1])
        else:
            cls_index = cfg.TRAIN.CLASSES[int(poses[i, 1])]
        cls_indexes.append(cls_index)

        qt = np.zeros((7, ), dtype=np.float32)
        qt[:3] = poses[i, 6:]
        qt[3:] = poses[i, 2:6]
        poses_all.append(qt)

    # rendering
    cfg.renderer.set_poses(poses_all)
    cfg.renderer.render(cls_indexes, image_tensor, seg_tensor, pc2_tensor=pcloud_tensor)
    image_tensor = image_tensor.flip(0)
    seg_tensor = seg_tensor.flip(0)
    pcloud_tensor = pcloud_tensor.flip(0)
    pcloud = pcloud_tensor[:,:,:3].cpu().numpy().reshape((-1, 3))

    im_label = seg_tensor.cpu().numpy()
    im_label = im_label[:, :, (2, 1, 0)] * 255
    im_label = np.round(im_label).astype(np.uint8)
    im_label = np.clip(im_label, 0, 255)
    im_label, im_label_all = dataset.process_label_image(im_label)

    # RGB to BGR order
    im_render = image_tensor.cpu().numpy()
    im_render = np.clip(im_render, 0, 1)
    im_render = im_render[:, :, :3] * 255
    im_render = im_render.astype(np.uint8)

    im_output = 0.4 * im[:,:,(2, 1, 0)].astype(np.float32) + 0.6 * im_render.astype(np.float32)

    return im_output.astype(np.uint8), im_label, pcloud


def initialize_poses(sample):
    pose_result = sample['poses_result'].numpy()
    roi_result = sample['rois_result'].numpy()
    if cfg.TEST.VISUALIZE:
        print('use posecnn result')
        print(pose_result)

    # construct poses target
    pose_est = np.zeros((0, 9), dtype=np.float32)
    roi_est = np.zeros((0, 7), dtype=np.float32)
    for i in range(pose_result.shape[0]):
        for j in range(pose_result.shape[1]):
            pose_result[i, j, 0] = i
            pose_est = np.concatenate((pose_est, pose_result[i, j, :].reshape(1, 9)), axis=0)
            roi_result[i, j, 0] = i
            roi_est = np.concatenate((roi_est, roi_result[i, j, :].reshape(1, 7)), axis=0)
    return pose_est, roi_est

# perturb target poses
def sample_poses(pose_tgt):
    pose_src = pose_tgt.copy()
    num = pose_tgt.shape[0]
    for i in range(num):
        euler = quat2euler(pose_tgt[i, 2:6])
        euler += cfg.TRAIN.SYN_STD_ROTATION * np.random.randn(3) * math.pi / 180.0
        pose_src[i, 2:6] = euler2quat(euler[0], euler[1], euler[2])
        pose_src[i, 6] += cfg.TRAIN.SYN_STD_TRANSLATION * np.random.randn(1)
        pose_src[i, 7] += cfg.TRAIN.SYN_STD_TRANSLATION * np.random.randn(1)
        pose_src[i, 8] += 5 * cfg.TRAIN.SYN_STD_TRANSLATION * np.random.randn(1)
    return pose_src


def process_sample(sample, poses_est, train_data):

    # image_blob is already in tensor GPU
    if cfg.MODE == 'TEST' and poses_est.shape[0] != 0:
        pose_blob = sample['poses_result'].numpy()
    else:
        pose_blob = sample['poses'].numpy()
    gt_boxes = sample['gt_boxes'].numpy()
    image_color_blob = sample['image_color']
    image_depth_blob = sample['image_depth']
    meta_data_blob = sample['meta_data'].numpy()
    label_blob = sample['label_blob'].numpy()
    extents = np.tile(sample['extents'][0, :, :].numpy(), (cfg.TRAIN.GPUNUM, 1, 1))
    points = np.tile(sample['points'][0, :, :, :].numpy(), (cfg.TRAIN.GPUNUM, 1, 1, 1))
    num_classes = points.shape[1]

    # construct poses target
    pose_tgt_blob = np.zeros((0, 9), dtype=np.float32)
    for i in range(pose_blob.shape[0]):
        for j in range(pose_blob.shape[1]):
            if pose_blob[i, j, -1] > 0:
                pose_blob[i, j, 0] = i
                pose_tgt_blob = np.concatenate((pose_tgt_blob, pose_blob[i, j, :].reshape(1, 9)), axis=0)

    # construct gt box
    gt_box_blob = np.zeros((0, 5), dtype=np.float32)
    pose_blob_gt = sample['poses'].numpy()
    for i in range(pose_blob_gt.shape[0]):
        for j in range(pose_blob_gt.shape[1]):
            if pose_blob_gt[i, j, -1] > 0:
                gt_box_blob = np.concatenate((gt_box_blob, gt_boxes[i, j, :].reshape(1, 5)), axis=0)

    num = pose_tgt_blob.shape[0]
    height = image_color_blob.shape[1]
    width = image_color_blob.shape[2]
    metadata = meta_data_blob[0, :]
    intrinsic_matrix = metadata[:9].reshape((3,3))
    weights_rot = np.zeros((num, num_classes * 4), dtype=np.float32)
    for i in range(num):
        cls = int(pose_tgt_blob[i, 1])
        weights_rot[i, 4*cls:4*cls+4] = 1.0

    if poses_est.shape[0] == 0:
        # sample source poses
        pose_src_blob = sample_poses(pose_tgt_blob)
    else:
        pose_src_blob = poses_est.copy()

    render_poses(intrinsic_matrix, label_blob, pose_tgt_blob, pose_src_blob, points[0], train_data)

    for i in range(num):
        image_id = int(pose_tgt_blob[i, 0])
        train_data['image_real_blob_color'][i].copy_(image_color_blob[image_id].permute(2, 0, 1))
        if pose_src_blob[i, 2] < 0:
            pose_src_blob[i, 2:6] = -1 * pose_src_blob[i, 2:6]

        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            train_data['image_real_blob_depth'][i].copy_(image_depth_blob[image_id].permute(2, 0, 1))

    vis_data = {'image': image_color_blob,
                'image_src': train_data['image_src_blob_color'],
                'image_tgt': train_data['image_tgt_blob_color'],
                'image_depth': image_depth_blob,
                'image_src_depth': train_data['image_src_blob_depth'],
                'image_tgt_depth': train_data['image_tgt_blob_depth'],
                'flow': train_data['flow_blob'],
                'intrinsic_matrix': intrinsic_matrix,
                'gt_boxes': gt_box_blob,
                'pose_src': pose_src_blob,
                'pose_tgt': pose_tgt_blob}

    # construct outputs
    train_data['input_blob_color'][:, :3, :, :] = train_data['image_src_blob_color']
    train_data['input_blob_color'][:, 3:6, :, :] = train_data['image_real_blob_color']
    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        train_data['input_blob_depth'][:, :3, :, :] = train_data['image_src_blob_depth']
        train_data['input_blob_depth'][:, 3:6, :, :] = train_data['image_real_blob_depth']

    return train_data['input_blob_color'][:num], train_data['input_blob_depth'][:num], \
           train_data['flow_blob'][:num], torch.from_numpy(pose_src_blob).contiguous(), \
           torch.from_numpy(pose_tgt_blob).contiguous(), torch.from_numpy(weights_rot).contiguous(), \
           torch.from_numpy(extents).contiguous(), torch.from_numpy(points).contiguous(), \
           train_data['affine_matrices'][:num], train_data['zoom_factor'][:num], vis_data


def train(train_loader, background_loader, network, optimizer, epoch, num_iterations):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_size = len(train_loader)
    enum_background = enumerate(background_loader)

    # declare tensors
    num = cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.MAX_OBJECT_PER_IMAGE
    height = cfg.TRAIN.SYN_HEIGHT
    width = cfg.TRAIN.SYN_WIDTH
    input_blob_color = torch.cuda.FloatTensor(num, 6, height, width).detach()
    image_real_blob_color = torch.cuda.FloatTensor(num, 3, height, width).detach()
    image_tgt_blob_color = torch.cuda.FloatTensor(num, 3, height, width).detach()
    image_src_blob_color = torch.cuda.FloatTensor(num, 3, height, width).detach()
    input_blob_depth = torch.cuda.FloatTensor(num, 6, height, width).detach()
    image_real_blob_depth = torch.cuda.FloatTensor(num, 3, height, width).detach()
    image_tgt_blob_depth = torch.cuda.FloatTensor(num, 3, height, width).detach()
    image_src_blob_depth = torch.cuda.FloatTensor(num, 3, height, width).detach()
    affine_matrices = torch.cuda.FloatTensor(num, 2, 3).detach()
    zoom_factor = torch.cuda.FloatTensor(num, 4).detach()
    flow_blob = torch.cuda.FloatTensor(num, 2, height, width).detach()
    pcloud_tgt_cuda = torch.cuda.FloatTensor(height, width, 3).detach()
    pcloud_src_cuda = torch.cuda.FloatTensor(height, width, 3).detach()
    flow_map_cuda = torch.cuda.FloatTensor(height, width, 2).detach()

    train_data = {'input_blob_color': input_blob_color,
                  'image_real_blob_color': image_real_blob_color,
                  'image_tgt_blob_color': image_tgt_blob_color,
                  'image_src_blob_color': image_src_blob_color,
                  'input_blob_depth': input_blob_depth,
                  'image_real_blob_depth': image_real_blob_depth,
                  'image_tgt_blob_depth': image_tgt_blob_depth,
                  'image_src_blob_depth': image_src_blob_depth,
                  'affine_matrices': affine_matrices,
                  'zoom_factor': zoom_factor,
                  'flow_blob': flow_blob,
                  'pcloud_tgt_cuda': pcloud_tgt_cuda,
                  'pcloud_src_cuda': pcloud_src_cuda,
                  'flow_map_cuda': flow_map_cuda}

    # switch to train mode
    network.train()
    cfg.ITERS = 0
    
    for i, sample in enumerate(train_loader):
        poses_est = np.zeros((0, 9), dtype=np.float32)

        # add background
        try:
            _, background = next(enum_background)
        except:
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        if sample['image_color'].size(0) != background['background_color'].size(0):
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        im_info = sample['im_info']
        mask = sample['mask']
        if cfg.INPUT == 'COLOR':
            background_color = background['background_color'].permute(0, 2, 3, 1)
            for j in range(sample['image_color'].size(0)):
                is_syn = im_info[j, -1]
                if is_syn or np.random.rand(1) > 0.5:
                    sample['image_color'][j] = mask[j] * sample['image_color'][j] + (1 - mask[j]) * background_color[j]
        elif cfg.INPUT == 'RGBD':
            background_color = background['background_color'].permute(0, 2, 3, 1)
            background_depth = background['background_depth'].permute(0, 2, 3, 1)
            for j in range(sample['image_color'].size(0)):
                is_syn = im_info[j, -1]
                if is_syn or np.random.rand(1) > 0.5:
                    sample['image_color'][j] = mask[j] * sample['image_color'][j] + (1 - mask[j]) * background_color[j]
                    sample['image_depth'][j] = mask[j] * sample['image_depth'][j] + (1 - mask[j]) * background_depth[j]

        # train multiple iterations
        for j in range(num_iterations):
            end = time.time()
            inputs, inputs_depth, flow, poses_src, poses_tgt, \
                weights_rot, extents, points, affine_matrices, zoom_factor, vdata = \
                process_sample(sample, poses_est, train_data)
            data_time.update(time.time() - end)

            # measure data loading time
            poses_src = poses_src.cuda().detach()
            poses_tgt = poses_tgt.cuda().detach()
            weights_rot = weights_rot.cuda().detach()
            extents = extents.cuda().detach()
            points = points.cuda().detach()

            # zoom in image
            grids = nn.functional.affine_grid(affine_matrices, inputs.size())
            input_zoom = nn.functional.grid_sample(inputs, grids).detach()
            if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
                input_zoom_depth = nn.functional.grid_sample(inputs_depth, grids).detach()

            # zoom in flow
            flow_zoom = nn.functional.grid_sample(flow, grids)
            for k in range(flow_zoom.shape[0]):
                flow_zoom[k, 0, :, :] /= affine_matrices[k, 0, 0] * 20.0
                flow_zoom[k, 1, :, :] /= affine_matrices[k, 1, 1] * 20.0

            if cfg.TRAIN.VISUALIZE:
                if cfg.INPUT == 'COLOR':
                    _vis_minibatch(inputs, input_zoom, flow_zoom, extents[0], vdata, 'COLOR')
                elif cfg.INPUT == 'DEPTH':
                    _vis_minibatch(inputs_depth, input_zoom_depth, flow_zoom, extents[0], vdata, 'DEPTH')
                elif cfg.INPUT == 'RGBD':
                    _vis_minibatch(inputs, input_zoom, flow_zoom, extents[0], vdata, 'COLOR')
                    _vis_minibatch(inputs_depth, input_zoom_depth, flow_zoom, extents[0], vdata, 'DEPTH')

            # compute output
            if cfg.INPUT == 'RGBD':
                output, loss_pose_tensor, quaternion_delta_var, translation_var \
                    = network(input_zoom, input_zoom_depth, weights_rot, poses_src, poses_tgt, extents, points, zoom_factor)
            else:
                if cfg.INPUT == 'COLOR':
                    x = input_zoom
                elif cfg.INPUT == 'DEPTH':
                    x = input_zoom_depth
                output, loss_pose_tensor, quaternion_delta_var, translation_var \
                    = network(x, weights_rot, poses_src, poses_tgt, extents, points, zoom_factor)

            # compose pose
            vdata_pose = vdata['pose_src']
            quaternion_delta = quaternion_delta_var.cpu().detach().numpy()
            translation = translation_var.cpu().detach().numpy()
            poses_est, error_rot, error_trans = _compute_pose_target(quaternion_delta, translation, vdata_pose, vdata['pose_tgt'])

            # losses
            loss_pose = torch.mean(loss_pose_tensor)
            loss_flow = 0.1 * multiscaleEPE(output, flow_zoom)
            flow2_EPE = realEPE(output[0], flow_zoom)
            loss = loss_pose + loss_flow

            # compute gradient and do optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)

            print('epoch: [%d/%d][%d/%d], iter %d, loss %.4f, l_pose %.4f (%.2f, %.2f), l_flow %.4f, lr %.6f, data time %.2f, batch time %.2f' \
                % (epoch, cfg.epochs, i, epoch_size, j+1, loss, loss_pose, error_rot, error_trans, loss_flow, \
                  optimizer.param_groups[0]['lr'], data_time.val, batch_time.val))

        cfg.ITERS += 1


def test(test_loader, background_loader, network, output_dir):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    epoch_size = len(test_loader)
    num_iterations = cfg.TEST.ITERNUM
    if background_loader is not None:
        enum_background = enumerate(background_loader)

    # declare tensors
    num = cfg.TEST.IMS_PER_BATCH * len(cfg.TRAIN.CLASSES)
    height = cfg.TRAIN.SYN_HEIGHT
    width = cfg.TRAIN.SYN_WIDTH

    input_blob_color = torch.cuda.FloatTensor(num, 6, height, width).detach()
    image_real_blob_color = torch.cuda.FloatTensor(num, 3, height, width).detach()
    image_tgt_blob_color = torch.cuda.FloatTensor(num, 3, height, width).detach()
    image_src_blob_color = torch.cuda.FloatTensor(num, 3, height, width).detach()
    input_blob_depth = torch.cuda.FloatTensor(num, 6, height, width).detach()
    image_real_blob_depth = torch.cuda.FloatTensor(num, 3, height, width).detach()
    image_tgt_blob_depth = torch.cuda.FloatTensor(num, 3, height, width).detach()
    image_src_blob_depth = torch.cuda.FloatTensor(num, 3, height, width).detach()
    affine_matrices = torch.cuda.FloatTensor(num, 2, 3).detach()
    zoom_factor = torch.cuda.FloatTensor(num, 4).detach()
    flow_blob = torch.cuda.FloatTensor(num, 2, height, width).detach()
    pcloud_tgt_cuda = torch.cuda.FloatTensor(height, width, 3).detach()
    pcloud_src_cuda = torch.cuda.FloatTensor(height, width, 3).detach()
    flow_map_cuda = torch.cuda.FloatTensor(height, width, 2).detach()

    test_data = {'input_blob_color': input_blob_color,
                 'image_real_blob_color': image_real_blob_color,
                 'image_tgt_blob_color': image_tgt_blob_color,
                 'image_src_blob_color': image_src_blob_color,
                 'input_blob_depth': input_blob_depth,
                 'image_real_blob_depth': image_real_blob_depth,
                 'image_tgt_blob_depth': image_tgt_blob_depth,
                 'image_src_blob_depth': image_src_blob_depth,
                 'affine_matrices': affine_matrices,
                 'zoom_factor': zoom_factor,
                 'flow_blob': flow_blob,
                 'pcloud_tgt_cuda': pcloud_tgt_cuda,
                 'pcloud_src_cuda': pcloud_src_cuda,
                 'flow_map_cuda': flow_map_cuda}

    # switch to test mode
    network.eval()
    cfg.ITERS = 0
    end = time.time()
    for i, sample in enumerate(test_loader):

        if 'is_testing' in sample and sample['is_testing'] == 0:
            continue

        result = []
        vis_data = []
        if cfg.TEST.SYNTHESIZE:
            # random initial poses
            poses_est = np.zeros((0, 9), dtype=np.float32)
            rois_est = np.zeros((0, 7), dtype=np.float32)
        else:
            # initialize poses from detection
            poses_est, rois_est = initialize_poses(sample)
            if poses_est.shape[0] == 0:
                continue

        # add background for testing on synthetic data
        if background_loader is not None:
            try:
                _, background = next(enum_background)
            except:
                enum_background = enumerate(background_loader)
                _, background = next(enum_background)

            if sample['image_color'].size(0) != background['background_color'].size(0):
                enum_background = enumerate(background_loader)
                _, background = next(enum_background)

            im_info = sample['im_info']
            mask = sample['mask']
            if cfg.INPUT == 'COLOR':
                background_color = background['background_color'].permute(0, 2, 3, 1)
                for j in range(sample['image_color'].size(0)):
                    is_syn = im_info[j, -1]
                    if is_syn or np.random.rand(1) > 0.5:
                        sample['image_color'][j] = mask[j] * sample['image_color'][j] + (1 - mask[j]) * background_color[j]
            elif cfg.INPUT == 'RGBD':
                background_color = background['background_color'].permute(0, 2, 3, 1)
                background_depth = background['background_depth'].permute(0, 2, 3, 1)
                for j in range(sample['image_color'].size(0)):
                    is_syn = im_info[j, -1]
                    if is_syn or np.random.rand(1) > 0.5:
                        sample['image_color'][j] = mask[j] * sample['image_color'][j] + (1 - mask[j]) * background_color[j]
                        sample['image_depth'][j] = mask[j] * sample['image_depth'][j] + (1 - mask[j]) * background_depth[j]

        for j in range(num_iterations):

            inputs, inputs_depth, flow, poses_src, poses_tgt, \
                weights_rot, extents, points, affine_matrices, zoom_factor, vdata = \
                process_sample(sample, poses_est, test_data)
            
            vis_data.append(vdata)
            if j == 0:
                poses_init = vdata['pose_src']

            poses_src = poses_src.cuda().detach()
            poses_tgt = poses_tgt.cuda().detach()
            weights_rot = weights_rot.cuda().detach()
            extents = extents.cuda().detach()
            points = points.cuda().detach()

            # zoom in image
            grids = nn.functional.affine_grid(affine_matrices, inputs.size())
            input_zoom = nn.functional.grid_sample(inputs, grids).detach()
            if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
                input_zoom_depth = nn.functional.grid_sample(inputs_depth, grids).detach()

            # compute output
            if cfg.INPUT == 'RGBD':
                quaternion_delta_var, translation_var \
                    = network(input_zoom, input_zoom_depth, weights_rot, poses_src, poses_tgt, extents, points, zoom_factor)
            else:
                if cfg.INPUT == 'COLOR':
                    x = input_zoom
                elif cfg.INPUT == 'DEPTH':
                    x = input_zoom_depth
                quaternion_delta_var, translation_var \
                    = network(x, weights_rot, poses_src, poses_tgt, extents, points, zoom_factor)

            # compose pose
            vdata_pose = vdata['pose_src']
            quaternion_delta = quaternion_delta_var.detach().cpu().numpy()
            translation = translation_var.detach().cpu().numpy()
            poses_est, error_rot, error_trans = _compute_pose_target(quaternion_delta, translation, vdata_pose, vdata['pose_tgt'])
            result.append(poses_est)

        pose = {'poses_init': poses_init, 'poses_est': result, 'rois': rois_est, \
                'poses_tgt': vdata['pose_tgt'], 'intrinsic_matrix': vdata['intrinsic_matrix']}

        if cfg.TEST.VISUALIZE:
            if cfg.INPUT == 'RGBD':
                _vis_test(result, vis_data, 'color')
                _vis_test(result, vis_data, 'depth')
            elif cfg.INPUT == 'COLOR':
                _vis_test(result, vis_data, 'color')
            else:
                _vis_test(result, vis_data, 'depth')
        else:
            # save result
            if 'video_id' in sample and 'image_id' in sample:
                filename = os.path.join(output_dir, sample['video_id'][0] + '_' + sample['image_id'][0] + '.mat')
            else:
                filename = os.path.join(output_dir, '%06d.mat' % i)
            print(filename)
            scipy.io.savemat(filename, pose, do_compression=True)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        cfg.ITERS += 1
        print('[%d/%d] %.4f' % (i, epoch_size, batch_time.val))

    filename = os.path.join(output_dir, 'results_deepim.mat')
    if os.path.exists(filename):
        os.remove(filename)


def test_image(network, dataset, im_color, im_depth, poses_est, test_data):

    # declare tensors
    batch_time = AverageMeter()
    num_iterations = cfg.TEST.ITERNUM
    num = poses_est.shape[0]
    height = cfg.TRAIN.SYN_HEIGHT
    width = cfg.TRAIN.SYN_WIDTH

    # construct sample
    im_tensor = torch.from_numpy(im_color).float() / 255.0
    im_tensor -= dataset._pixel_mean
    im_cuda_color = im_tensor.cuda()

    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        pcloud = backproject(im_depth, dataset._intrinsic_matrix, is_reshape=True)
        im_cuda_depth = torch.from_numpy(pcloud).cuda().float()
    else:
        im_cuda_depth = im_cuda_color.clone().detach()
        
    # construct the meta data
    K = dataset._intrinsic_matrix
    Kinv = np.linalg.pinv(K)
    meta_data_blob = np.zeros(18, dtype=np.float32)
    meta_data_blob[0:9] = K.flatten()
    meta_data_blob[9:18] = Kinv.flatten()
    label_blob = np.zeros((dataset.num_classes, height, width), dtype=np.float32)
    gt_boxes = np.zeros((num, 5), dtype=np.float32)
    im_info = np.array([im_color.shape[0], im_color.shape[1], cfg.TRAIN.SCALES_BASE[0]], dtype=np.float32)

    sample = {'image_color': im_cuda_color.unsqueeze(0),
              'image_depth': im_cuda_depth.unsqueeze(0),
              'meta_data': torch.from_numpy(meta_data_blob[np.newaxis,:]),
              'label_blob': torch.from_numpy(label_blob[np.newaxis,:]),
              'poses': torch.from_numpy(poses_est[np.newaxis,:]),
              'extents': torch.from_numpy(dataset._extents[np.newaxis,:]),
              'points': torch.from_numpy(dataset._point_blob[np.newaxis,:]),
              'gt_boxes': torch.from_numpy(gt_boxes[np.newaxis,:]),
              'poses_result': torch.from_numpy(poses_est[np.newaxis,:]),
              'im_info': torch.from_numpy(im_info[np.newaxis,:])}

    # switch to test mode
    network.eval()
    end = time.time()
    result = []
    vis_data = []

    for j in range(num_iterations):

        inputs, inputs_depth, flow, poses_src, poses_tgt, \
            weights_rot, extents, points, affine_matrices, \
            zoom_factor, vdata = process_sample(sample, poses_est, test_data)

        vis_data.append(vdata)
        if j == 0:
            poses_init = vdata['pose_src']

        poses_src = poses_src.cuda().detach()
        poses_tgt = poses_tgt.cuda().detach()
        weights_rot = weights_rot.cuda().detach()
        extents = extents.cuda().detach()
        points = points.cuda().detach()

        # zoom in image
        grids = nn.functional.affine_grid(affine_matrices, inputs.size())
        input_zoom = nn.functional.grid_sample(inputs, grids).detach()
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            input_zoom_depth = nn.functional.grid_sample(inputs_depth, grids).detach()

        # compute output
        if cfg.INPUT == 'RGBD':
            quaternion_delta_var, translation_var \
                = network(input_zoom, input_zoom_depth, weights_rot, poses_src, poses_tgt, extents, points, zoom_factor)
        else:
            if cfg.INPUT == 'COLOR':
                x = input_zoom
            elif cfg.INPUT == 'DEPTH':
                x = input_zoom_depth
            quaternion_delta_var, translation_var \
                = network(x, weights_rot, poses_src, poses_tgt, extents, points, zoom_factor)

        # compose pose
        vdata_pose = vdata['pose_src']
        quaternion_delta = quaternion_delta_var.detach().cpu().numpy()
        translation = translation_var.detach().cpu().numpy()
        poses_est, error_rot, error_trans = _compute_pose_target(quaternion_delta, translation, vdata_pose, vdata['pose_tgt'])
        result.append(poses_est)
        
    pose = {'poses_init': poses_init, 'poses_est': result, 'poses_tgt': vdata['pose_tgt'], 'intrinsic_matrix': vdata['intrinsic_matrix']}

    # render result
    im_pose, im_label, pcloud = render_image(dataset, im_color, result[-1])

    if cfg.TEST.VISUALIZE:
        if cfg.INPUT == 'RGBD':
            _vis_test(result, vis_data, 'color')
            _vis_test(result, vis_data, 'depth')
        elif cfg.INPUT == 'COLOR':
            _vis_test(result, vis_data, 'color')
        else:
            _vis_test(result, vis_data, 'depth')

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    print('time %.4f' % (batch_time.val))

    return im_pose, pose


# backproject pixels into 3D points in camera's coordinate system
def backproject(depth_cv, intrinsic_matrix, is_reshape=False):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)
    if is_reshape:
        index = np.where(np.isnan(X))
        X[index[0], index[1]] = 0
        return np.array(X).transpose().reshape((height, width, 3))
    else:
        return np.array(X).transpose()


def refine_pose(im_label, im_depth, pose_result, pcloud):

    # backprojection
    intrinsic_matrix = pose_result['intrinsic_matrix']
    dpoints = backproject(im_depth, intrinsic_matrix)

    # renderer
    poses = pose_result['poses_est'][-1]
    num = poses.shape[0]
    width = im_depth.shape[1]
    height = im_depth.shape[0]
    im_label = im_label.reshape((width * height, ))

    # refine pose
    for i in range(num):
        cls = int(poses[i, 1]) + 1
        index = np.where((im_label == cls) & np.isfinite(dpoints[:, 0]) & (pcloud[:, 0] != 0))[0]
        if len(index) > 10:
            T = np.mean(dpoints[index, :] - pcloud[index, :], axis=0)
            poses[i, 8] += T[2]
        else:
            print('no pose refinement')

    pose_result['poses_est'][-1] = poses
    return pose_result


def overlay_image(dataset, im, poses):

    im = im[:, :, (2, 1, 0)]
    classes = dataset._classes
    class_colors = dataset._class_colors
    points = dataset._points_all
    intrinsic_matrix = dataset._intrinsic_matrix
    height = im.shape[0]
    width = im.shape[1]

    for j in range(poses.shape[0]):
        cls = int(poses[j, 1])
        print(classes[cls])
        if cls >= 0:

            # extract 3D points
            x3d = np.ones((4, points.shape[1]), dtype=np.float32)
            x3d[0, :] = points[cls,:,0]
            x3d[1, :] = points[cls,:,1]
            x3d[2, :] = points[cls,:,2]

            # projection
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(poses[j, 2:6])
            RT[:, 3] = poses[j, 6:]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x = np.round(np.divide(x2d[0, :], x2d[2, :]))
            y = np.round(np.divide(x2d[1, :], x2d[2, :]))
            index = np.where((x >= 0) & (x < width) & (y >= 0) & (y < height))[0]
            x = x[index].astype(np.int32)
            y = y[index].astype(np.int32)
            im[y, x, 0] = class_colors[cls][0]
            im[y, x, 1] = class_colors[cls][1]
            im[y, x, 2] = class_colors[cls][2]

    return im


def _compute_pose_target(quaternion_delta, translation, poses_src, poses_gt):
    poses_tgt = poses_src.copy()
    num = poses_src.shape[0]
    errors_rot = np.zeros((num, ), dtype=np.float32)
    errors_trans = np.zeros((num, ), dtype=np.float32)
    for i in range(poses_src.shape[0]):
        cls = int(poses_src[i, 1])
        poses_tgt[i, 2:6] = mat2quat(np.dot(quat2mat(quaternion_delta[i, 4*cls:4*cls+4]), quat2mat(poses_src[i, 2:6])))
        poses_tgt[i, 6:] = translation[i, 3*cls:3*cls+3]
        # compute pose errors
        errors_rot[i] = np.arccos(2 * np.power(np.dot(poses_tgt[i, 2:6], poses_gt[i, 2:6]), 2) - 1) * 180.0 / np.pi
        errors_trans[i] = np.linalg.norm(poses_tgt[i, 6:] - poses_gt[i, 6:]) * 100
    return poses_tgt, np.mean(errors_rot), np.mean(errors_trans)

def _get_bb3D(extent):
    bb = np.zeros((3, 8), dtype=np.float32)
    
    xHalf = extent[0] * 0.5
    yHalf = extent[1] * 0.5
    zHalf = extent[2] * 0.5
    
    bb[:, 0] = [xHalf, yHalf, zHalf]
    bb[:, 1] = [-xHalf, yHalf, zHalf]
    bb[:, 2] = [xHalf, -yHalf, zHalf]
    bb[:, 3] = [-xHalf, -yHalf, zHalf]
    bb[:, 4] = [xHalf, yHalf, -zHalf]
    bb[:, 5] = [-xHalf, yHalf, -zHalf]
    bb[:, 6] = [xHalf, -yHalf, -zHalf]
    bb[:, 7] = [-xHalf, -yHalf, -zHalf]
    
    return bb

def _vis_test(result, vis_data, input_type):

    num_iter = len(result)
    pose_blob = vis_data[0]['pose_tgt']
    num_obj = pose_blob.shape[0]
    if input_type == 'color':
        im_blob = vis_data[0]['image'].cpu().numpy()
    else:
        im_blob = vis_data[0]['image_depth'].cpu().numpy()

    import matplotlib.pyplot as plt
    for j in range(num_obj):
        image_id = int(pose_blob[j, 0])

        fig = plt.figure()

        # show input image
        im_input = im_blob[image_id, :, :, :].copy()
        im_input *= 255
        im_input += cfg.PIXEL_MEANS
        im_input = im_input[:, :, (2, 1, 0)]
        im_input = im_input.astype(np.uint8)
        ax = fig.add_subplot(3, num_iter, 1)
        plt.imshow(im_input)
        ax.set_title('input image')

        for i in range(num_iter):
            poses_est = result[i]

            intrinsic_matrix = vis_data[i]['intrinsic_matrix']
            poses_src = vis_data[i]['pose_src']
            if input_type == 'color':
                image_src_blob = vis_data[i]['image_src'].permute(0, 2, 3, 1).cpu().numpy()
                image_tgt_blob = vis_data[i]['image_tgt'].permute(0, 2, 3, 1).cpu().numpy()
            else:
                image_src_blob = vis_data[i]['image_src_depth'].permute(0, 2, 3, 1).cpu().numpy()
                image_tgt_blob = vis_data[i]['image_tgt_depth'].permute(0, 2, 3, 1).cpu().numpy()
            poses_tgt = vis_data[i]['pose_tgt']

            height = image_src_blob.shape[1]
            width = image_src_blob.shape[2]

            # images in BGR order
            num = poses_est.shape[0]
            images_est = np.zeros((num, height, width, 3), dtype=np.float32)
            render_one_poses(height, width, intrinsic_matrix, poses_est, images_est)
            images_est = images_est.astype(np.uint8)
    
            # compute error
            R_est = quat2mat(poses_est[j, 2:6])
            R_src = quat2mat(poses_src[j, 2:6])
            R_tgt = quat2mat(poses_tgt[j, 2:6])
            error_rot_src = re(R_src, R_tgt)
            error_rot_est = re(R_est, R_tgt)

            T_est = poses_est[j, 6:]
            T_src = poses_src[j, 6:]
            T_tgt = poses_tgt[j, 6:]
            error_tran_src = te(T_src, T_tgt)
            error_tran_est = te(T_est, T_tgt)

            # show rendered images
            im = image_src_blob[j, :, :, :].copy()
            im *= 255
            im += cfg.PIXEL_MEANS
            im = np.clip(im, 0, 255)
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
            ax = fig.add_subplot(3, num_iter, num_iter + 1 + i)
            ax.set_title('source iter %d (rot %.2f, tran %.4f)' % (i+1, error_rot_src, error_tran_src)) 
            plt.imshow(im)

            if i == 0:
                im = image_tgt_blob[j, :, :, :3].copy()
                im *= 255
                im += cfg.PIXEL_MEANS
                im = np.clip(im, 0, 255)
                im = im[:, :, (2, 1, 0)]
                im = im.astype(np.uint8)

                im_output = 0.7 * im.astype(np.float32) + 0.3 * im_input.astype(np.float32)
                im_output = np.clip(im_output, 0, 255)
                im_output = im_output.astype(np.uint8)

                ax = fig.add_subplot(3, num_iter, 2)
                ax.set_title('target image') 
                plt.imshow(im_output)

            # show estimated image
            im = images_est[j, :, :, :3].copy()
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)

            im_output = 0.7 * im.astype(np.float32) + 0.3 * im_input.astype(np.float32)
            im_output = np.clip(im_output, 0, 255)
            im_output = im_output.astype(np.uint8)

            ax = fig.add_subplot(3, num_iter, 2 * num_iter + 1 + i)
            ax.set_title('estimate iter %d (rot %.2f, tran %.4f)' % (i+1, error_rot_est, error_tran_est)) 
            plt.imshow(im_output)

        plt.show()


def convert_to_image(im_blob):
    return np.clip(255 * im_blob, 0, 255).astype(np.uint8)


def _vis_minibatch(input_var, input_zoom, flow_zoom, extents, vdata, input_format='COLOR'):

    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    pose_blob = vdata['pose_tgt']
    pose_src = vdata['pose_src']
    if input_format == 'COLOR':
        im_blob = vdata['image'].cpu().numpy()
        image_src_blob = vdata['image_src'].permute(0, 2, 3, 1).cpu().numpy()
        image_tgt_blob = vdata['image_tgt'].permute(0, 2, 3, 1).cpu().numpy()
    else:
        im_blob = vdata['image_depth'].cpu().numpy()
        image_src_blob = vdata['image_src_depth'].permute(0, 2, 3, 1).cpu().numpy()
        image_tgt_blob = vdata['image_tgt_depth'].permute(0, 2, 3, 1).cpu().numpy()

    intrinsic_matrix = vdata['intrinsic_matrix']
    gt_boxes = vdata['gt_boxes']
    flow_blob = vdata['flow'].permute(0, 2, 3, 1).cpu().numpy()

    for j in range(pose_blob.shape[0]):
        image_id = int(pose_blob[j, 0])
        fig = plt.figure()

        # compute pose distances
        error_rot = np.arccos(2 * np.power(np.dot(pose_blob[j, 2:6], pose_src[j, 2:6]), 2) - 1) * 180.0 / np.pi
        error_trans = np.linalg.norm(pose_blob[j, 6:] - pose_src[j, 6:]) * 100
        s = 'rot %.2f, trans %.2f' % (error_rot, error_trans)

        # show image
        im = im_blob[image_id, :, :, :].copy()
        if input_format == 'COLOR':
            im *= 255
            im += cfg.PIXEL_MEANS
            im = np.clip(im, 0, 255)
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
        else:
            im = im[:, :, 2]
        ax = fig.add_subplot(4, 3, 1)
        plt.imshow(im)
        ax.set_title(s)
        plt.axis('off')

        # show projection box
        class_id = int(pose_blob[j, 1])
        bb3d = _get_bb3D(extents[class_id, :])
        x3d = np.ones((4, 8), dtype=np.float32)
        x3d[0:3, :] = bb3d
            
        RT = np.zeros((3, 4), dtype=np.float32)
        RT[:3, :3] = quat2mat(pose_blob[j, 2:6])
        RT[:, 3] = pose_blob[j, 6:]
        x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

        x1 = np.min(x2d[0, :])
        x2 = np.max(x2d[0, :])
        y1 = np.min(x2d[1, :])
        y2 = np.max(x2d[1, :])
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3))

        # show gt box
        ax = fig.add_subplot(4, 3, 2)
        plt.imshow(im)
        ax.set_title('gt box')
        x1 = gt_boxes[j, 0]
        y1 = gt_boxes[j, 1]
        x2 = gt_boxes[j, 2]
        y2 = gt_boxes[j, 3]
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3))
        plt.axis('off')

        # show rendered images
        if input_format == 'COLOR':
            im = image_src_blob[j, :, :, :].copy()
            im *= 255
            im += cfg.PIXEL_MEANS
            im = np.clip(im, 0, 255)
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
        else:
            im = image_src_blob[j, :, :, 2].copy()
        ax = fig.add_subplot(4, 3, 4)
        ax.set_title('source image render') 
        plt.imshow(im)
        plt.axis('off')

        if input_format == 'COLOR':
            im = image_tgt_blob[j, :, :, :3].copy()
            im *= 255
            im += cfg.PIXEL_MEANS
            im = np.clip(im, 0, 255)
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
        else:
            im = image_tgt_blob[j, :, :, 2].copy()
        ax = fig.add_subplot(4, 3, 5)
        ax.set_title('target image render') 
        plt.imshow(im)
        plt.axis('off')

        # show zoomed images
        if input_format == 'COLOR':
            im = input_zoom[j, :3, :, :].cpu().numpy()
            im = 255 * np.transpose(im, (1, 2, 0))
            im += cfg.PIXEL_MEANS
            im = np.clip(im, 0, 255)
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
        else:
            im = input_zoom[j, 2, :, :].cpu().numpy()
        ax = fig.add_subplot(4, 3, 3)
        ax.set_title('zoomed source image') 
        plt.imshow(im)
        plt.axis('off')

        if input_format == 'COLOR':
            im = input_zoom[j, 3:6, :, :].cpu().numpy()
            im = 255 * np.transpose(im, (1, 2, 0))
            im += cfg.PIXEL_MEANS
            im = np.clip(im, 0, 255)
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
        else:
            im = input_zoom[j, 5, :, :].cpu().numpy()
        ax = fig.add_subplot(4, 3, 6)
        ax.set_title('zoomed target image') 
        plt.imshow(im)
        plt.axis('off')

        # show flow
        flow = flow_blob[j, :, :, 0]
        ax = fig.add_subplot(4, 3, 7)
        ax.set_title('flow x') 
        plt.imshow(flow)
        plt.axis('off')

        flow = flow_blob[j, :, :, 1]
        ax = fig.add_subplot(4, 3, 8)
        ax.set_title('flow y') 
        plt.imshow(flow)
        plt.axis('off')

        flow_image = sintel_compute_color(flow_blob[j, :, :, :])
        ax = fig.add_subplot(4, 3, 9)
        ax.set_title('flow image') 
        plt.imshow(flow_image)
        plt.axis('off')

        # show zoomed flow
        flow = flow_zoom[j, 0, :, :].cpu().numpy()
        ax = fig.add_subplot(4, 3, 10)
        ax.set_title('zoomed flow x') 
        plt.imshow(flow)
        plt.axis('off')

        flow = flow_zoom[j, 1, :, :].cpu().numpy()
        ax = fig.add_subplot(4, 3, 11)
        ax.set_title('zoomed flow y') 
        plt.imshow(flow)
        plt.axis('off')

        flow = flow_zoom[j, :, :, :].cpu().numpy()
        flow_image = sintel_compute_color(np.transpose(flow, (1, 2, 0)))
        ax = fig.add_subplot(4, 3, 12)
        ax.set_title('zoomed flow image') 
        plt.imshow(flow_image)
        plt.axis('off')

        plt.show()
