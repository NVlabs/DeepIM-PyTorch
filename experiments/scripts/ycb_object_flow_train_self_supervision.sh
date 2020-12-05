#!/bin/bash

set -x
set -e

time ./tools/train_net.py \
  --network flownets \
  --pretrained output/ycb_object/ycb_object_train/flownets_ycb_object_20objects_color_epoch_20.checkpoint.pth \
  --dataset ycb_self_supervision_all_ycb \
  --cfg experiments/cfgs/ycb_object_flow_self_supervision.yml \
  --solver sgd \
  --epochs 10
