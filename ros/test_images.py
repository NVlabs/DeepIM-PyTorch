#!/usr/bin/env python

# --------------------------------------------------------
# DeepIM
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a DeepIM on images"""

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
from fcn.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file
from datasets.factory import get_dataset
import networks
import rospy
from listener import ImageListener

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a DeepIM network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--instance', dest='instance_id', help='DeepIM instance id to use',
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
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

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

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.device = torch.device('cuda:{:d}'.format(0))
    print 'GPU device {:d}'.format(args.gpu_id)
    cfg.gpu_id = args.gpu_id
    cfg.instance_id = args.instance_id

    # dataset
    cfg.classes = cfg.TEST.CLASSES
    cfg.MODE = 'TEST'
    cfg.TEST.SYNTHESIZE = False
    dataset = get_dataset(args.dataset_name)

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](dataset.num_classes, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[0]).cuda(device=cfg.device)
    cudnn.benchmark = True

    # image listener
    network.eval()
    listener = ImageListener(network, dataset)

    while not rospy.is_shutdown():       
       listener.run_network()
