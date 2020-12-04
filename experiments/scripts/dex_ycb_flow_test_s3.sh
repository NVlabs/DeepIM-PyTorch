#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_net.py --gpu $1 \
  --network flownets \
  --pretrained output/dex_ycb/dex_ycb_s3_train/flownets_dex_ycb_all_epoch_$2.checkpoint.pth \
  --dataset dex_ycb_s3_test \
  --cfg experiments/cfgs/dex_ycb_flow.yml
