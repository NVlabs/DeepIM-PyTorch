#!/bin/bash

set -x
set -e

time ./tools/train_net.py \
  --network flownets \
  --pretrained data/checkpoints/flownets_EPE1.951.pth \
  --dataset ycb_object_train \
  --cfg experiments/cfgs/ycb_object_flow.yml \
  --solver sgd \
  --epochs 20
