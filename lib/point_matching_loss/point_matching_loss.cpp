// Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
// This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
// text can be found in LICENSE.md

#include <torch/torch.h>

#include <vector>

// CUDA forward declarations
// point matching loss (pml)

std::vector<at::Tensor> pml_cuda_forward(
    at::Tensor bottom_rotations,
    at::Tensor bottom_translations,
    at::Tensor poses_src,
    at::Tensor poses_tgt,
    at::Tensor extents,
    at::Tensor points);

std::vector<at::Tensor> pml_cuda_backward(
    at::Tensor grad_loss,
    at::Tensor bottom_diff_rotation,
    at::Tensor bottom_diff_translation);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> pml_forward(
    at::Tensor rotations,
    at::Tensor translations,
    at::Tensor poses_src,
    at::Tensor poses_tgt,
    at::Tensor extents,
    at::Tensor points) {
  CHECK_INPUT(rotations);
  CHECK_INPUT(translations);
  CHECK_INPUT(poses_src);
  CHECK_INPUT(poses_tgt);
  CHECK_INPUT(extents);
  CHECK_INPUT(points);

  return pml_cuda_forward(rotations, translations, poses_src, poses_tgt, extents, points);
}

std::vector<at::Tensor> pml_backward(
    at::Tensor grad_loss,
    at::Tensor bottom_diff_rotation,
    at::Tensor bottom_diff_translation) {
  CHECK_INPUT(grad_loss);
  CHECK_INPUT(bottom_diff_rotation);
  CHECK_INPUT(bottom_diff_translation);

  return pml_cuda_backward(grad_loss, bottom_diff_rotation, bottom_diff_translation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pml_forward, "pml forward (CUDA)");
  m.def("backward", &pml_backward, "pml backward (CUDA)");
}
