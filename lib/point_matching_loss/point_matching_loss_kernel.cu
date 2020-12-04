// Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
// This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
// text can be found in LICENSE.md

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define POSE_CHANNELS 9

// poses_src: (batch_size, 9)
template <typename Dtype>
__global__ void AveragedistanceForward(const int nthreads, 
    const Dtype* bottom_rotations, const Dtype* bottom_translations,
    const Dtype* poses_src, const Dtype* poses_tgt, const Dtype* extents, const Dtype* points, 
    const int batch_size, const int num_classes, const int num_points, 
    Dtype* rotations, Dtype* losses, Dtype* diffs_rotation, Dtype* diffs_translation)
{
  CUDA_1D_KERNEL_LOOP(index_thread, nthreads) 
  {
    // batch index
    int n = index_thread / num_points;
    int p = index_thread % num_points;

    // find the class label and pose of this object
    int index_cls = int(poses_src[n * POSE_CHANNELS + 1]);
    int ind;
    Dtype s, u, v, w;

    // gt quaternion target
    int index = n * POSE_CHANNELS + 2;
    s = poses_tgt[index + 0];
    u = poses_tgt[index + 1];
    v = poses_tgt[index + 2];
    w = poses_tgt[index + 3];

    // gt rotation matrix target
    ind = n * num_points * 7 * 9 + p * 7 * 9 + 0;
    rotations[ind + 0] = s * s + u * u - v * v - w * w;
    rotations[ind + 1] = 2 * (u * v - s * w);
    rotations[ind + 2] = 2 * (u * w + s * v);
    rotations[ind + 3] = 2 * (u * v + s * w);
    rotations[ind + 4] = s * s - u * u + v * v - w * w;
    rotations[ind + 5] = 2 * (v * w - s * u);
    rotations[ind + 6] = 2 * (u * w - s * v);
    rotations[ind + 7] = 2 * (v * w + s * u);
    rotations[ind + 8] = s * s - u * u - v * v + w * w;

    // gt quaternion source
    s = poses_src[index + 0];
    u = poses_src[index + 1];
    v = poses_src[index + 2];
    w = poses_src[index + 3];

    // gt rotation matrix source
    ind = n * num_points * 7 * 9 + p * 7 * 9 + 9;
    rotations[ind + 0] = s * s + u * u - v * v - w * w;
    rotations[ind + 1] = 2 * (u * v - s * w);
    rotations[ind + 2] = 2 * (u * w + s * v);
    rotations[ind + 3] = 2 * (u * v + s * w);
    rotations[ind + 4] = s * s - u * u + v * v - w * w;
    rotations[ind + 5] = 2 * (v * w - s * u);
    rotations[ind + 6] = 2 * (u * w - s * v);
    rotations[ind + 7] = 2 * (v * w + s * u);
    rotations[ind + 8] = s * s - u * u - v * v + w * w;

    // predicted quaternion
    index = n * 4 * num_classes + 4 * index_cls;
    s = bottom_rotations[index + 0];
    u = bottom_rotations[index + 1];
    v = bottom_rotations[index + 2];
    w = bottom_rotations[index + 3];

    // predicted rotation matrix
    ind = n * num_points * 7 * 9 + p * 7 * 9 + 18;
    rotations[ind + 0] = s * s + u * u - v * v - w * w;
    rotations[ind + 1] = 2 * (u * v - s * w);
    rotations[ind + 2] = 2 * (u * w + s * v);
    rotations[ind + 3] = 2 * (u * v + s * w);
    rotations[ind + 4] = s * s - u * u + v * v - w * w;
    rotations[ind + 5] = 2 * (v * w - s * u);
    rotations[ind + 6] = 2 * (u * w - s * v);
    rotations[ind + 7] = 2 * (v * w + s * u);
    rotations[ind + 8] = s * s - u * u - v * v + w * w;

    // derivatives of Ru to quaternion
    ind = n * num_points * 7 * 9 + p * 7 * 9 + 27;
    rotations[ind + 0] = 2 * s;
    rotations[ind + 1] = -2 * w;
    rotations[ind + 2] = 2 * v;
    rotations[ind + 3] = 2 * w;
    rotations[ind + 4] = 2 * s;
    rotations[ind + 5] = -2 * u;
    rotations[ind + 6] = -2 * v;
    rotations[ind + 7] = 2 * u;
    rotations[ind + 8] = 2 * s;

    ind = n * num_points * 7 * 9 + p * 7 * 9 + 36;
    rotations[ind + 0] = 2 * u;
    rotations[ind + 1] = 2 * v;
    rotations[ind + 2] = 2 * w;
    rotations[ind + 3] = 2 * v;
    rotations[ind + 4] = -2 * u;
    rotations[ind + 5] = -2 * s;
    rotations[ind + 6] = 2 * w;
    rotations[ind + 7] = 2 * s;
    rotations[ind + 8] = -2 * u;

    ind = n * num_points * 7 * 9 + p * 7 * 9 + 45;
    rotations[ind + 0] = -2 * v;
    rotations[ind + 1] = 2 * u;
    rotations[ind + 2] = 2 * s;
    rotations[ind + 3] = 2 * u;
    rotations[ind + 4] = 2 * v;
    rotations[ind + 5] = 2 * w;
    rotations[ind + 6] = -2 * s;
    rotations[ind + 7] = 2 * w;
    rotations[ind + 8] = -2 * v;

    ind = n * num_points * 7 * 9 + p * 7 * 9 + 54;
    rotations[ind + 0] = -2 * w;
    rotations[ind + 1] = -2 * s;
    rotations[ind + 2] = 2 * u;
    rotations[ind + 3] = 2 * s;
    rotations[ind + 4] = -2 * w;
    rotations[ind + 5] = 2 * v;
    rotations[ind + 6] = 2 * u;
    rotations[ind + 7] = 2 * v;
    rotations[ind + 8] = 2 * w;

    // for the point
    index = index_cls * num_points * 3 + p * 3;
    ind = n * num_points * 7 * 9 + p * 7 * 9;

    // weight for the point
    Dtype weight = -1;
    for (int j = 0; j < 3; j++)
    {
      if (extents[index_cls * 3 + j] > weight)
        weight = extents[index_cls * 3 + j];
    }
    weight = 10.0 / weight;

    // rotate the point using the source pose
    Dtype x1_0 = rotations[ind + 9 + 0] * points[index + 0] + rotations[ind + 9 + 1] * points[index + 1] + rotations[ind + 9 + 2] * points[index + 2];
    Dtype y1_0 = rotations[ind + 9 + 3] * points[index + 0] + rotations[ind + 9 + 4] * points[index + 1] + rotations[ind + 9 + 5] * points[index + 2];
    Dtype z1_0 = rotations[ind + 9 + 6] * points[index + 0] + rotations[ind + 9 + 7] * points[index + 1] + rotations[ind + 9 + 8] * points[index + 2];

    x1_0 *= weight;
    y1_0 *= weight;
    z1_0 *= weight;

    // rotate the point again using the estimated delta rotation
    int index_tran = n * 3 * num_classes + 3 * index_cls;
    Dtype x1 = rotations[ind + 18 + 0] * x1_0 + rotations[ind + 18 + 1] * y1_0 + rotations[ind + 18 + 2] * z1_0 
             + weight * bottom_translations[index_tran + 0];
    Dtype y1 = rotations[ind + 18 + 3] * x1_0 + rotations[ind + 18 + 4] * y1_0 + rotations[ind + 18 + 5] * z1_0 
             + weight * bottom_translations[index_tran + 1];
    Dtype z1 = rotations[ind + 18 + 6] * x1_0 + rotations[ind + 18 + 7] * y1_0 + rotations[ind + 18 + 8] * z1_0 
             + weight * bottom_translations[index_tran + 2];

    // rotate and translate the point using the target pose
    Dtype x2 = weight * (rotations[ind + 0] * points[index + 0] + rotations[ind + 1] * points[index + 1] + rotations[ind + 2] * points[index + 2])
             + weight * poses_tgt[n * POSE_CHANNELS + 6];
    Dtype y2 = weight * (rotations[ind + 3] * points[index + 0] + rotations[ind + 4] * points[index + 1] + rotations[ind + 5] * points[index + 2])
             + weight * poses_tgt[n * POSE_CHANNELS + 7];
    Dtype z2 = weight * (rotations[ind + 6] * points[index + 0] + rotations[ind + 7] * points[index + 1] + rotations[ind + 8] * points[index + 2])
             + weight * poses_tgt[n * POSE_CHANNELS + 8];

    // smooth l1 loss
    Dtype distance = 0;
    int index_diff = n * num_points * 4 * num_classes + p * 4 * num_classes + 4 * index_cls;
    int index_diff_t = n * num_points * 3 * num_classes + p * 3 * num_classes + 3 * index_cls;
    for (int j = 0; j < 3; j++)
    {
      Dtype diff, df;
      if (j == 0)
        diff = x1 - x2;
      else if (j == 1)
        diff = y1 - y2;
      else
        diff = z1 - z2;

      if (fabs(diff) < 1)
      {
        distance += 0.5 * diff * diff;
        df = diff;
      }
      else
      {
        distance += fabs(diff) - 0.5;
        if (diff > 0)
          df = 1.0;
        else
          df = -1.0;
      }

      for (int k = 0; k < 3; k++)
      {
        Dtype dp;
        if (k == 0)
          dp = x1_0;
        else if (k == 1)
          dp = y1_0;
        else
          dp = z1_0;

        ind = n * num_points * 7 * 9 + p * 7 * 9 + 27;
        diffs_rotation[index_diff + 0] += df * dp * rotations[ind + j * 3 + k] / (batch_size * num_points);
        ind = n * num_points * 7 * 9 + p * 7 * 9 + 36;
        diffs_rotation[index_diff + 1] += df * dp * rotations[ind + j * 3 + k] / (batch_size * num_points);
        ind = n * num_points * 7 * 9 + p * 7 * 9 + 45;
        diffs_rotation[index_diff + 2] += df * dp * rotations[ind + j * 3 + k] / (batch_size * num_points);
        ind = n * num_points * 7 * 9 + p * 7 * 9 + 54;
        diffs_rotation[index_diff + 3] += df * dp * rotations[ind + j * 3 + k] / (batch_size * num_points);
      }

      diffs_translation[index_diff_t + j] = weight * df / (batch_size * num_points);
    }

    losses[index_thread] = distance / (batch_size * num_points);
  }
}



template <typename Dtype>
__global__ void sum_losses_gradients(const int nthreads, const Dtype* losses, const Dtype* poses_src,
    const Dtype* diffs_rotation, const Dtype* diffs_translation, const int num_classes, const int num_points, 
    Dtype* losses_batch, Dtype* bottom_diff_rotation, Dtype* bottom_diff_translation) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int index_cls = int(poses_src[index * POSE_CHANNELS + 1]);
    losses_batch[index] = 0;

    bottom_diff_rotation[4 * index * num_classes + 4 * index_cls + 0] = 0;
    bottom_diff_rotation[4 * index * num_classes + 4 * index_cls + 1] = 0;
    bottom_diff_rotation[4 * index * num_classes + 4 * index_cls + 2] = 0;
    bottom_diff_rotation[4 * index * num_classes + 4 * index_cls + 3] = 0;

    bottom_diff_translation[3 * index * num_classes + 3 * index_cls + 0] = 0;
    bottom_diff_translation[3 * index * num_classes + 3 * index_cls + 1] = 0;
    bottom_diff_translation[3 * index * num_classes + 3 * index_cls + 2] = 0;

    for (int p = 0; p < num_points; p++)
    {
      losses_batch[index] += losses[index * num_points + p];

      int index_diff = index * num_points * 4 * num_classes + p * 4 * num_classes + 4 * index_cls;
      bottom_diff_rotation[4 * index * num_classes + 4 * index_cls + 0] += diffs_rotation[index_diff + 0];
      bottom_diff_rotation[4 * index * num_classes + 4 * index_cls + 1] += diffs_rotation[index_diff + 1];
      bottom_diff_rotation[4 * index * num_classes + 4 * index_cls + 2] += diffs_rotation[index_diff + 2];
      bottom_diff_rotation[4 * index * num_classes + 4 * index_cls + 3] += diffs_rotation[index_diff + 3];

      index_diff = index * num_points * 3 * num_classes + p * 3 * num_classes + 3 * index_cls;
      bottom_diff_translation[3 * index * num_classes + 3 * index_cls + 0] += diffs_translation[index_diff + 0];
      bottom_diff_translation[3 * index * num_classes + 3 * index_cls + 1] += diffs_translation[index_diff + 1];
      bottom_diff_translation[3 * index * num_classes + 3 * index_cls + 2] += diffs_translation[index_diff + 2];
    }
  }
}


std::vector<at::Tensor> pml_cuda_forward(
    at::Tensor bottom_rotations,
    at::Tensor bottom_translations,
    at::Tensor poses_src,
    at::Tensor poses_tgt,
    at::Tensor extents,
    at::Tensor points) 
{
  // run kernels
  const int kThreadsPerBlock = 512;
  int output_size;

  // temp losses
  const int batch_size = bottom_rotations.size(0);
  const int num_classes = points.size(1);
  const int num_points = points.size(2); 

  auto losses = at::zeros({batch_size, num_points}, points.options());
  auto losses_batch = at::zeros({batch_size}, points.options());
  auto loss = at::zeros({1}, points.options());

  // temp diffs
  auto diffs_rotation = at::zeros({batch_size, num_points, 4 * num_classes}, points.options());
  auto bottom_diff_rotation = at::zeros({batch_size, 4 * num_classes}, points.options());

  auto diffs_translation = at::zeros({batch_size, num_points, 3 * num_classes}, points.options());
  auto bottom_diff_translation = at::zeros({batch_size, 3 * num_classes}, points.options());

  // temp rotations
  auto rots = at::zeros({batch_size, num_points, 7 * 9}, points.options());

  AT_DISPATCH_FLOATING_TYPES(points.type(), "pml_forward_cuda", ([&] {

    // compute the losses and gradients
    output_size = batch_size * num_points;
    AveragedistanceForward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        output_size,
        bottom_rotations.data<scalar_t>(),
        bottom_translations.data<scalar_t>(),
        poses_src.data<scalar_t>(),
        poses_tgt.data<scalar_t>(),
        extents.data<scalar_t>(),
        points.data<scalar_t>(),
        batch_size,
        num_classes,
        num_points,
        rots.data<scalar_t>(),
        losses.data<scalar_t>(),
        diffs_rotation.data<scalar_t>(),
        diffs_translation.data<scalar_t>());

    // sum the diffs
    output_size = batch_size;
    sum_losses_gradients<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        output_size, 
        losses.data<scalar_t>(),
        poses_src.data<scalar_t>(),
        diffs_rotation.data<scalar_t>(),
        diffs_translation.data<scalar_t>(), 
        num_classes,
        num_points,
        losses_batch.data<scalar_t>(),
        bottom_diff_rotation.data<scalar_t>(),
        bottom_diff_translation.data<scalar_t>());

    // sum the loss
    thrust::device_ptr<float> losses_ptr(losses_batch.data<float>());
    float loss_value = thrust::reduce(losses_ptr, losses_ptr + batch_size);
    cudaMemcpy(loss.data<float>(), &loss_value, sizeof(float), cudaMemcpyHostToDevice);
  }));

  return {loss, bottom_diff_rotation, bottom_diff_translation};
}


template <typename Dtype>
__global__ void AveragedistanceBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_diff, Dtype* output) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    output[index] = top_diff[0] * bottom_diff[index];
  }
}


std::vector<at::Tensor> pml_cuda_backward(
    at::Tensor grad_loss,
    at::Tensor bottom_diff_rotation,
    at::Tensor bottom_diff_translation)
{
  const int kThreadsPerBlock = 512;
  int output_size;
  const int batch_size = bottom_diff_rotation.size(0);
  const int num_classes = bottom_diff_rotation.size(1) / 4;

  auto grad_rotation = at::zeros({batch_size, 4 * num_classes}, bottom_diff_rotation.options());
  auto grad_translation = at::zeros({batch_size, 3 * num_classes}, bottom_diff_translation.options());

  AT_DISPATCH_FLOATING_TYPES(grad_loss.type(), "pml_backward_cuda", ([&] {

    output_size = batch_size * 4 * num_classes;
    AveragedistanceBackward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        output_size, 
        grad_loss.data<scalar_t>(),
        bottom_diff_rotation.data<scalar_t>(),
        grad_rotation.data<scalar_t>());

    output_size = batch_size * 3 * num_classes;
    AveragedistanceBackward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        output_size, 
        grad_loss.data<scalar_t>(),
        bottom_diff_translation.data<scalar_t>(),
        grad_translation.data<scalar_t>());

  }));

  return {grad_rotation, grad_translation};
}
