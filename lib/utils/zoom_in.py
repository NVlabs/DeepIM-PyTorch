# --------------------------------------------------------
# FCN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import numpy as np
import cv2
from fcn.config import cfg

def zoom_images(image_blob, image_real, image_imgn, image_flow, pose_src, intrinsic_matrix):

    batch_size = image_real.shape[0]
    height = image_real.shape[1]
    width = image_real.shape[2]
    ratio = float(height) / float(width)

    image_zoom = np.zeros((batch_size, height ,width, 3), dtype=np.float32)
    image_real_zoom = np.zeros((batch_size, height ,width, 3), dtype=np.float32)
    image_imgn_zoom = np.zeros((batch_size, height ,width, 3), dtype=np.float32)
    flow_zoom = np.zeros((batch_size, height ,width, 2), dtype=np.float32)
    mask_real = np.zeros((batch_size, height ,width, 1), dtype=np.float32)
    mask_imgn = np.zeros((batch_size, height ,width, 1), dtype=np.float32)
    zoom_factor = np.zeros((batch_size, 4), dtype=np.float32)

    for i in range(batch_size):

        # real image
        nz_y, nz_x = np.where(image_real[i, :, :, 0] > 0)        
        obj_real_start_x = np.min(nz_x)
        obj_real_end_x = np.max(nz_x)
        obj_real_start_y = np.min(nz_y)
        obj_real_end_y = np.max(nz_y)
        obj_real_c_x = (obj_real_start_x + obj_real_end_x) * 0.5
        obj_real_c_y = (obj_real_start_y + obj_real_end_y) * 0.5

        # rendered image
        nz_y, nz_x = np.where(image_imgn[i, :, :, 0] > 0)
        obj_imgn_start_x = np.min(nz_x)
        obj_imgn_end_x = np.max(nz_x)
        obj_imgn_start_y = np.min(nz_y)
        obj_imgn_end_y = np.max(nz_y)
        obj_imgn_c = np.dot(intrinsic_matrix, pose_src[i, 6:])
        zoom_c_x = obj_imgn_c[0] / obj_imgn_c[2]
        zoom_c_y = obj_imgn_c[1] / obj_imgn_c[2]

        # mask region
        left_dist = max(zoom_c_x - obj_imgn_start_x, zoom_c_x - obj_real_start_x)
        right_dist = max(obj_imgn_end_x - zoom_c_x, obj_real_end_x - zoom_c_x)
        up_dist = max(zoom_c_y - obj_imgn_start_y, zoom_c_y - obj_real_start_y)
        down_dist = max(obj_real_end_y - zoom_c_y, obj_imgn_end_y - zoom_c_y)

        # crop_height = np.max([up_dist, down_dist])
        # crop_width = np.max([left_dist, right_dist])
        crop_height = np.max([ratio * right_dist, ratio * left_dist, up_dist, down_dist]) * 2
        crop_width = crop_height / ratio

        # affine transformation
        x1 = zoom_c_x - crop_width / 2
        x2 = zoom_c_x + crop_width / 2
        y1 = zoom_c_y - crop_height / 2
        y2 = zoom_c_y + crop_height / 2

        pts1 = np.float32([[x1, y1], [x1, y2], [x2, y1]])
        pts2 = np.float32([[0, 0], [0, height], [width, 0]])
        affine_matrix = cv2.getAffineTransform(pts1, pts2)

        idx = int(pose_src[i, 0])
        image_zoom[i, :, :, :] = cv2.warpAffine(image_blob[idx, :, :, :], affine_matrix, (width, height))
        image_real_zoom[i, :, :, :] = cv2.warpAffine(image_real[i, :, :, :], affine_matrix, (width, height))
        image_imgn_zoom[i, :, :, :] = cv2.warpAffine(image_imgn[i, :, :, :], affine_matrix, (width, height))
        flow_zoom[i, :, :, :] =  cv2.warpAffine(image_flow[i, :, :, :], affine_matrix, (width, height))
        flow_zoom[i, :, :, 0] *= affine_matrix[0, 0] 
        flow_zoom[i, :, :, 1] *= affine_matrix[1, 1]

        # construct masks
        nz_y, nz_x = np.where(image_imgn_zoom[i, :, :, 0] > 0)
        x1 = int(np.min(nz_x))
        x2 = int(np.max(nz_x))
        y1 = int(np.min(nz_y))
        y2 = int(np.max(nz_y))
        mask_real[i, y1:y2, x1:x2, :] = 1.0
        mask_imgn[i, nz_y, nz_x, :] = 1.0

        # image_zoom[i, :, :, :] -= cfg.PIXEL_MEANS
        # image_real_zoom[i, :, :, :] -= cfg.PIXEL_MEANS
        # image_imgn_zoom[i, :, :, :] -= cfg.PIXEL_MEANS

        zoom_factor[i, 0] = affine_matrix[0, 0]
        zoom_factor[i, 1] = affine_matrix[1, 1]
        zoom_factor[i, 2] = affine_matrix[0, 2]
        zoom_factor[i, 3] = affine_matrix[1, 2]

    return image_zoom, image_real_zoom, image_imgn_zoom, flow_zoom, mask_real, mask_imgn, zoom_factor
