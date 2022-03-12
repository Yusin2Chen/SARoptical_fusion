from typing import Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometric import rotate
from .misc import _extract_device_dtype


def normalize_kernel2d(input):
    """
        Normalizes both derivative and smoothing kernel.

    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))

    norm = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def get_box_kernel2d(kernel_size):
    """
        Utility function that returns a box filter.

    """

    kx = float(kernel_size[0])
    ky = float(kernel_size[1])

    scale = torch.tensor(1.) / torch.tensor([kx * ky])

    tmp_kernel = torch.ones(1, kernel_size[0], kernel_size[1])

    return scale.to(tmp_kernel.dtype) * tmp_kernel


def box_blur(input, kernel_size, border_type='reflect', normalized=True):
    """
        Blurs an image using the box filter.

    """
    kernel = get_box_kernel2d(kernel_size)

    if normalized:
        kernel = normalize_kernel2d(kernel)

    return filter2D(input, kernel, border_type)


def gaussian(window_size, sigma):
    """
        Computer Gaussian.

    """

    x = torch.arange(window_size) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / (2 * sigma ** 2)))

    return gauss / gauss.sum()


def get_gaussian_kernel1d(kernel_size, sigma, force_even=False):
    """
        Function that returns Gaussian filter coefficients.

    """
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )
    window_1d = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(kernel_size, sigma, force_even=False):
    """
        Get Gaussian filter matrix coefficients.
    """

    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )

    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma

    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)

    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)

    kernel_2d = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())

    return kernel_2d


def gaussian_blur2d(input, kernel_size, sigma, border_type='reflect'):
    """
        Blurs a tensor using a Gaussian filter.
    """
    kernel = torch.unsqueeze(get_gaussian_kernel2d(kernel_size, sigma), dim=0)

    return filter2D(input, kernel, border_type)


def _compute_padding(kernel_size):
    """
        Computes padding tuple.

    """

    assert len(kernel_size) >= 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp

    return out_padding


def get_motion_kernel2d(kernel_size, angle, direction=0., mode='nearest'):
    """
        Return 2D motion blur filter.

    """

    device, dtype = _extract_device_dtype([
        angle if isinstance(angle, torch.Tensor) else None,
        direction if isinstance(direction, torch.Tensor) else None,
    ])

    if not isinstance(any(kernel_size), int) or any(kernel_size % 2 == 0) or any(kernel_size < 3):
        raise TypeError("ksize must be an odd integer >= than 3")

    if angle.dim() == 0:
        angle = angle.unsqueeze(0)
    assert angle.dim() == 1, f"angle must be a 1-dim tensor. Got {angle}."

    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor([direction], device=device, dtype=dtype)

    if direction.dim() == 0:
        direction = direction.unsqueeze(0)
    assert direction.dim() == 1, f"direction must be a 1-dim tensor. Got {direction}."

    assert direction.size(0) == angle.size(0), \
        f"direction and angle must have the same length. Got {direction} and {angle}."

    kernel_tuple = (kernel_size, kernel_size)

    # direction from [-1, 1] to [0, 1] range
    direction = (torch.clamp(direction, -1., 1.) + 1.) / 2.
    # kernel = torch.zeros((direction.size(0), *kernel_tuple), device=device, dtype=dtype)

    # Element-wise linspace
    # kernel[:, kernel_size // 2, :] = torch.stack(
    #     [(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], dim=-1)
    # Alternatively
    # m = ((1 - 2 * direction)[:, None].repeat(1, kernel_size) / (kernel_size - 1))
    # kernel[:, kernel_size // 2, :] = direction[:, None].repeat(1, kernel_size) + m * torch.arange(0, kernel_size)
    k = torch.stack([(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], dim=-1)

    kernel = torch.nn.functional.pad(k[:, None], [0, 0, kernel_size // 2, kernel_size // 2, 0, 0])

    assert kernel.shape == torch.Size([direction.size(0), *kernel_tuple])
    kernel = kernel.unsqueeze(1)
    # rotate (counterclockwise) kernel by given angle

    kernel = rotate(kernel, angle, interpolation=mode, align_corners=True)
    kernel = kernel[:, 0]
    kernel = kernel / kernel.sum(dim=(1, 2), keepdim=True)

    return kernel


def filter2D(input, kernel, border_type='reflect', normalized=False):
    """
        Convolve a tensor with a 2d kernel.

    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input border_type is not torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input border_type is not torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3 and kernel.shape[0] != 1:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]

    padding_shape = _compute_padding([height, width])

    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-2), input_pad.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    return output.view(b, c, h, w)


def motion_blur(input, kernel_size, angle, direction, border_type='constant', mode='nearest'):
    """
        Apply Motion Blur.

    """
    assert border_type in ["constant", "reflect", "replicate", "circular"]
    kernel = get_motion_kernel2d(kernel_size, angle, direction, mode)
    return filter2D(input, kernel, border_type)