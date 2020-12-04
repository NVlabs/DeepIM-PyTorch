#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_net.py --gpu $1 \
  --network flownets \
  --pretrained output/ycb_object/ycb_object_train/flownets_ycb_object_20objects_color_epoch_$2.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_flow.yml
