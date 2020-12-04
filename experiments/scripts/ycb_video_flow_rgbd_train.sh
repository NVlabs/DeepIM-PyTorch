#!/bin/bash

set -x
set -e

time ./tools/train_net.py \
  --network flownets_rgbd \
  --pretrained data/checkpoints/flownets_EPE1.951.pth \
  --dataset ycb_video_train \
  --dataset_background background_sunrgbd \
  --cfg experiments/cfgs/ycb_video_flow_rgbd.yml \
  --solver sgd \
  --epochs 20
