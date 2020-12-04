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
