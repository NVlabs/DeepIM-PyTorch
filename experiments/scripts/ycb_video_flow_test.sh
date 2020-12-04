#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_net.py --gpu 0 \
  --network flownets \
  --pretrained output/ycb_video/ycb_video_train/flownets_ycb_video_all_epoch_$2.checkpoint.pth \
  --dataset ycb_video_keyframe \
  --cfg experiments/cfgs/ycb_video_flow.yml
