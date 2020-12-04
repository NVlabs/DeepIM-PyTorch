#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./ros/test_images.py --gpu $1 \
  --network flownets \
  --pretrained output/ycb_object/ycb_object_train/flownets_ycb_object_20objects_color_epoch_5.checkpoint.pth \
  --dataset ycb_object_train \
  --cfg experiments/cfgs/ycb_object_flow.yml
