#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./ros/test_images.py --gpu $1 \
  --network flownets \
  --pretrained data/checkpoints/ycb_object/flownets_ycb_object_20objects_color_self_supervision_epoch_10.checkpoint.pth \
  --dataset ycb_object_train \
  --cfg experiments/cfgs/ycb_object_flow.yml
