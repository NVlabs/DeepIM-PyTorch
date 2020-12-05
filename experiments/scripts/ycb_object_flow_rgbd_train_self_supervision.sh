#!/bin/bash

set -x
set -e

time ./tools/train_net.py \
  --network flownets_rgbd \
  --pretrained output/ycb_object/ycb_object_train/flownets_ycb_object_20objects_rgbd_epoch_20.checkpoint.pth \
  --dataset ycb_self_supervision_all_ycb \
  --dataset_background background_sunrgbd \
  --cfg experiments/cfgs/ycb_object_flow_rgbd_self_supervision.yml \
  --solver sgd \
  --epochs 10
