# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import math
from torch import nn
from torch.autograd import Function
import torch
import pml_cuda


class PMLossFunction(Function):
    @staticmethod
    def forward(ctx, rotations, translations, poses_src, poses_tgt, extents, points):
        outputs = pml_cuda.forward(rotations, translations, poses_src, poses_tgt, extents, points)
        loss = outputs[0]
        variables = outputs[1:]
        ctx.save_for_backward(*variables)

        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        outputs = pml_cuda.backward(grad_loss.contiguous(), *ctx.saved_variables)
        d_rotation, d_translation = outputs

        return d_rotation, d_translation, None, None, None, None


class PMLoss(nn.Module):
    def __init__(self):
        super(PMLoss, self).__init__()

    def forward(self, rotations, translations, poses_src, poses_tgt, extents, points):
        return PMLossFunction.apply(rotations, translations, poses_src, poses_tgt, extents, points)
