# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pml',
    ext_modules=[
        CUDAExtension('pml_cuda', [
            'point_matching_loss.cpp',
            'point_matching_loss_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
