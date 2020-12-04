#!/bin/bash

set -x
set -e

time ./tools/train_net.py \
  --network flownets \
  --pretrained data/checkpoints/flownets_EPE1.951.pth \
  --dataset dex_ycb_s0_train \
  --cfg experiments/cfgs/dex_ycb_flow.yml \
  --solver sgd \
  --epochs 20
