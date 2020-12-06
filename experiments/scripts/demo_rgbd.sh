#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

time ./tools/test_images.py --gpu 0 \
  --imgdir data/demo/ \
  --meta data/demo/meta.yml \
  --color *color.png \
  --network flownets_rgbd \
  --pretrained data/checkpoints/ycb_object/flownets_ycb_object_20objects_rgbd_self_supervision_epoch_10.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_flow_rgbd.yml
