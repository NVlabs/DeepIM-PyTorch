#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a DeepIM network on an image database."""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np

import _init_paths
from fcn.train_test import test
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_dataset
import networks
from ycb_renderer import YCBRenderer

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a DeepIM network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--cad', dest='cad_name',
                        help='name of the CAD file',
                        default=None, type=str)
    parser.add_argument('--pose', dest='pose_name',
                        help='name of the pose files',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)
    parser.add_argument('--dataset_background', dest='dataset_background_name',
                        help='background dataset to train on',
                        default='background_nvidia', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize and not cfg.TEST.VISUALIZE:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    print('GPU device {:d}'.format(args.gpu_id))
    
    cfg.classes = cfg.TEST.CLASSES
    # prepare dataset
    if cfg.TEST.VISUALIZE:
        shuffle = True
    else:
        shuffle = False 
    cfg.MODE = 'TEST'
    
    dataset = get_dataset(args.dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=shuffle, num_workers=0)
    print('Use dataset `{:s}` for training'.format(dataset.name))

    # background dataset
    if cfg.TEST.SYNTHESIZE:
        if cfg.TRAIN.SYN_BACKGROUND_SPECIFIC:
            background_dataset = get_dataset(args.dataset_background_name)
        else:
            background_dataset = get_dataset('background_coco')
        background_loader = torch.utils.data.DataLoader(background_dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                                        shuffle=True, num_workers=4)
    else:
        background_loader = None

    cfg.TEST.MODEL = args.pretrained.split('/')[-1]
    output_dir = get_output_dir(dataset, None)
    output_dir = os.path.join(output_dir, cfg.TEST.MODEL)
    print('Output will be saved to `{:s}`'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('loading 3D models')
    cfg.renderer = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, render_marker=False, gpu_id=args.gpu_id)
    if cfg.TEST.SYNTHESIZE:
        cfg.renderer.load_objects(dataset.model_mesh_paths, dataset.model_texture_paths, dataset.model_colors)
        print(dataset.model_mesh_paths)
    else:
        cfg.renderer.load_objects(dataset.model_mesh_paths_target, dataset.model_texture_paths_target, dataset.model_colors_target)
        print(dataset.model_mesh_paths_target)
    cfg.renderer.set_camera_default()

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()
    # make sure the model is loaded
    network = networks.__dict__[args.network_name](len(cfg.TRAIN.CLASSES), network_data).cuda()
    network = torch.nn.DataParallel(network).cuda()
    cudnn.benchmark = True

    # test network
    test(dataloader, background_loader, network, output_dir)

    # evaluation
    dataset.evaluation(output_dir)
