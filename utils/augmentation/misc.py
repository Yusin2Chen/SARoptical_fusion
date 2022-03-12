import torch
from typing import cast

from torch.distributions import Uniform, Beta, Bernoulli

pi = torch.tensor(3.14159265358979323846)


def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor.
    """
    if not isinstance(obj, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(obj)))


def _adapted_uniform(shape, low, high):
    r"""The uniform sampling function that accepts 'same_on_batch'.
    """
    low = torch.as_tensor(low, device=low.device, dtype=low.dtype)
    high = torch.as_tensor(high, device=high.device, dtype=high.dtype)

    dist = Uniform(low, high)

    return dist.rsample(shape)


def rad2deg(tensor):
    check_is_tensor(tensor)

    return 180. * tensor / pi.to(tensor.device).type(tensor.dtype)


def deg2rad(tensor):

    check_is_tensor(tensor)

    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.


def bbox_generator(x_start, y_start, width, height):
    """Generate 2D bounding boxes according to the provided start coords, width and height.

    """
    assert x_start.shape == y_start.shape and x_start.dim() in [0, 1], \
        f"`x_start` and `y_start` must be a scalar or (B,). Got {x_start}, {y_start}."
    assert width.shape == height.shape and width.dim() in [0, 1], \
        f"`width` and `height` must be a scalar or (B,). Got {width}, {height}."
    assert x_start.dtype == y_start.dtype == width.dtype == height.dtype, (
        "All tensors must be in the same dtype. Got "
        f"`x_start`({x_start.dtype}), `y_start`({x_start.dtype}), `width`({width.dtype}), `height`({height.dtype})."
    )
    assert x_start.device == y_start.device == width.device == height.device, (
        "All tensors must be in the same device. Got "
        f"`x_start`({x_start.device}), `y_start`({x_start.device}), `width`({width.device}), `height`({height.device})."
    )

    bbox = torch.tensor([[
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]], device=x_start.device, dtype=x_start.dtype).repeat(1 if x_start.dim() == 0 else len(x_start), 1, 1)

    bbox[:, :, 0] += x_start.view(-1, 1)
    bbox[:, :, 1] += y_start.view(-1, 1)
    bbox[:, 1, 0] += width - 1
    bbox[:, 2, 0] += width - 1
    bbox[:, 2, 1] += height - 1
    bbox[:, 3, 1] += height - 1

    return bbox


def _compute_tensor_center(tensor):

    assert 2 <= len(tensor.shape) <= 4, f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}."

    height, width = tensor.shape[-2:]

    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2

    center: torch.Tensor = torch.tensor([center_x, center_y], device=tensor.device, dtype=tensor.dtype)
    return center


def angle_to_rotation_matrix(angle):
    r"""Create a rotation matrix out of angles in degrees.

    """
    ang_rad = deg2rad(angle)
    cos_a = torch.cos(ang_rad)
    sin_a = torch.sin(ang_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)


def get_rotation_matrix2d(center, angle, scale):
    r"""Calculates an affine matrix of 2D rotation.
    """
    if not isinstance(center, torch.Tensor):
        raise TypeError("Input center type is not a torch.Tensor. Got {}"
                        .format(type(center)))
    if not isinstance(angle, torch.Tensor):
        raise TypeError("Input angle type is not a torch.Tensor. Got {}"
                        .format(type(angle)))
    if not isinstance(scale, torch.Tensor):
        raise TypeError("Input scale type is not a torch.Tensor. Got {}"
                        .format(type(scale)))

    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError("Input center must be a Bx2 tensor. Got {}"
                         .format(center.shape))
    if not len(angle.shape) == 1:
        raise ValueError("Input angle must be a B tensor. Got {}"
                         .format(angle.shape))
    if not (len(scale.shape) == 2 and scale.shape[1] == 2):
        raise ValueError("Input scale must be a Bx2 tensor. Got {}"
                         .format(scale.shape))
    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Got center {}, angle {} and scale {}"
                         .format(center.shape, angle.shape, scale.shape))
    if not (center.device == angle.device == scale.device) or not (center.dtype == angle.dtype == scale.dtype):
        raise ValueError("Inputs must have same device Got center ({}, {}), angle ({}, {}) and scale ({}, {})"
                         .format(center.device, center.dtype, angle.device, angle.dtype, scale.device, scale.dtype))

    # convert angle and apply scale
    rotation_matrix = angle_to_rotation_matrix(angle)

    scaling_matrix = torch.zeros((2, 2),
                                 device=center.device,
                                 dtype=center.dtype).fill_diagonal_(1).repeat(rotation_matrix.size(0), 1, 1)

    scaling_matrix = scaling_matrix * scale.unsqueeze(dim=2).repeat(1, 1, 2)
    scaled_rotation = rotation_matrix @ scaling_matrix

    alpha = scaled_rotation[:, 0, 0]
    beta = scaled_rotation[:, 0, 1]

    # unpack the center to x, y coordinates
    x = center[..., 0]
    y = center[..., 1]

    # create output tensor
    batch_size = center.shape[0]

    one = torch.tensor(1., device=center.device, dtype=center.dtype)

    M = torch.zeros(batch_size, 2, 3, device=center.device, dtype=center.dtype)

    M[..., 0:2, 0:2] = scaled_rotation
    M[..., 0, 2] = (one - alpha) * x - beta * y
    M[..., 1, 2] = beta * x + (one - alpha) * y

    return M


def _convert_affinematrix_to_homography_impl(A):
    H = torch.nn.functional.pad(A, [0, 0, 0, 1], "constant", value=0.)
    H[..., -1, -1] += 1.0
    return H


def convert_affinematrix_to_homography(A):
    r"""Function that converts batch of affine matrices from [Bx2x3] to [Bx3x3].
    """
    check_is_tensor(A)

    if not (len(A.shape) == 3 and A.shape[-2:] == (2, 3)):
        raise ValueError("Input matrix must be a Bx2x3 tensor. Got {}"
                         .format(A.shape))

    return _convert_affinematrix_to_homography_impl(A)


def get_shear_matrix2d(center, sx=None, sy=None):
    r"""Composes shear matrix Bx4x4 from the components.
    """
    sx = torch.tensor([0.]).repeat(center.size(0)) if sx is None else sx
    sy = torch.tensor([0.]).repeat(center.size(0)) if sy is None else sy

    x, y = torch.split(center, 1, dim=-1)
    x, y = x.view(-1), y.view(-1)

    sx_tan = torch.tan(sx)  # type: ignore
    sy_tan = torch.tan(sy)  # type: ignore

    ones = torch.ones_like(sx)  # type: ignore

    shear_mat = torch.stack([
        ones,               -sx_tan,                    sx_tan * y,
        -sy_tan,    ones + sx_tan * sy_tan,     sy_tan * (sx_tan * y + x)
    ], dim=-1).view(-1, 2, 3)

    shear_mat = convert_affinematrix_to_homography(shear_mat)
    return shear_mat

def _compute_rotation_matrix(angle, center):
    """Computes a pure affine rotation matrix."""
    scale: torch.Tensor = torch.ones_like(center)
    matrix: torch.Tensor = get_rotation_matrix2d(center.to(angle.device), angle, scale.to(angle.device))
    return matrix


def create_meshgrid(height, width, normalized_coordinates=True, device=torch.device('cpu')):
    """Generates a coordinate grid for an image.
    When the flag `normalized_coordinates` is set to True, the grid is
    """
    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)

    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2

    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def convert_points_from_homogeneous(points, eps=1e-8):
    r"""Function that converts points from homogeneous to Euclidean space.
    """
    check_is_tensor(points)

    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    # we check for points at infinity
    z_vec: torch.Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask: torch.Tensor = torch.abs(z_vec) > eps
    scale: torch.Tensor = torch.ones_like(z_vec).masked_scatter_(
        mask, torch.tensor(1.0).to(points.device) / z_vec[mask])

    return scale * points[..., :-1]


def convert_points_to_homogeneous(points):
    r"""Function that converts points from Euclidean to homogeneous space.

    """
    check_is_tensor(points)

    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)


def transform_points(trans_01, points_1):
    r"""Function that applies transformations to a set of points.

    """

    check_is_tensor(trans_01)
    check_is_tensor(points_1)

    if not (trans_01.device == points_1.device and trans_01.dtype == points_1.dtype):
        raise TypeError(
            "Tensor must be in the same device and dtype. "
            f"Got trans_01 with ({trans_01.dtype}, {points_1.dtype}) and "
            f"points_1 with ({points_1.dtype}, {points_1.dtype})")

    if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
        raise ValueError("Input batch size must be the same for both tensors or 1")

    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError("Last input dimensions must differ by one unit")

    # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
    shape_inp = list(points_1.shape)
    points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
    trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])

    # We expand trans_01 to match the dimensions needed for bmm
    trans_01 = torch.repeat_interleave(trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0)

    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1

    # transform coordinates
    points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)

    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD

    # reshape to the input shape
    shape_inp[-2] = points_0.shape[-2]
    shape_inp[-1] = points_0.shape[-1]

    points_0 = points_0.reshape(shape_inp)

    return points_0


def rgb_to_hsv(image):
    r"""Convert an image from RGB to HSV.
    The image data is assumed to be in the range of (0, 1).

    """
    check_is_tensor(image)

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    # The first or last occurance is not guarenteed before 1.6.0
    # https://github.com/pytorch/pytorch/issues/20414
    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc = image.min(-3)[0]

    v = maxc  # brightness

    deltac = maxc - minc
    s = deltac / (v + 1e-31)

    # avoid division by zero
    deltac = torch.where(deltac == 0, torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype), deltac)

    maxc_tmp = maxc.unsqueeze(-3) - image
    rc = maxc_tmp[..., 0, :, :]
    gc = maxc_tmp[..., 1, :, :]
    bc = maxc_tmp[..., 2, :, :]

    h = torch.stack([
        bc - gc,
        2.0 * deltac + rc - bc,
        4.0 * deltac + gc - rc,
    ], dim=-3)

    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac

    h = (h / 6.0) % 1.0

    h = 2 * pi * h

    return torch.stack([h, s, v], dim=-3)


def hsv_to_rgb(image):
    r"""Convert an image from HSV to RGB.
    The image data is assumed to be in the range of (0, 1).

    """
    check_is_tensor(image)

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    h = image[..., 0, :, :] / (2 * pi)
    s = image[..., 1, :, :]
    v = image[..., 2, :, :]

    hi = torch.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    one = torch.tensor(1.).to(image.device)
    p = v * (one - s)
    q = v * (one - f * s)
    t = v * (one - (one - f) * s)

    hi = hi.long()
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((
        v, q, p, p, t, v,
        t, v, v, q, p, p,
        p, p, t, v, v, q,
    ), dim=-3)
    out = torch.gather(out, -3, indices)

    return out


def _to_bchw(tensor):
    """Converts a PyTorch tensor image to BCHW format.
    """
    check_is_tensor(tensor)

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError(f"Input size must be a two, three or four dimensional tensor. Got {tensor.shape}")

    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    return tensor


def _to_bcdhw(tensor, color_channel_num=None):
    """Converts a PyTorch tensor image to BCHW format.

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) > 5 or len(tensor.shape) < 3:
        raise ValueError(f"Input size must be a three, four or five dimensional tensor. Got {tensor.shape}")

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)

    return tensor


def rgb_to_grayscale(image):
    r"""Convert a RGB image to grayscale version of image.
    The image data is assumed to be in the range of (0, 1).
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r = image[..., 0:1, :, :]
    g = image[..., 1:2, :, :]
    b = image[..., 2:3, :, :]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


def bgr_to_grayscale(image):
    r"""Convert a BGR image to grayscale.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    image_rgb = bgr_to_rgb(image)
    gray = rgb_to_grayscale(image_rgb)
    return gray


def rgb_to_bgr(image):
    r"""Convert a RGB image to BGR.
    Args:
        image (torch.Tensor): RGB Image to be converted to BGRof of shape :math:`(*,3,H,W)`.
    Returns:
        torch.Tensor: BGR version of the image with shape of shape :math:`(*,3,H,W)`.
    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_bgr(input) # 2x3x4x5
    """
    return bgr_to_rgb(image)


def bgr_to_rgb(image):
    r"""Convert a BGR image to RGB.
    Args:
        image (torch.Tensor): BGR Image to be converted to BGR of shape :math:`(*,3,H,W)`.
    Returns:
        torch.Tensor: RGB version of the image with shape of shape :math:`(*,3,H,W)`.
    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = bgr_to_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}"
                         .format(image.shape))

    # flip image channels
    out: torch.Tensor = image.flip(-3)
    return out


def rgb_to_rgba(image, alpha_val):
    r"""Convert an image from RGB to RGBA.
    Args:
        image (torch.Tensor): RGB Image to be converted to RGBA of shape :math:`(*,3,H,W)`.
        alpha_val (float, torch.Tensor): A float number for the alpha value or a tensor
          of shape :math:`(*,1,H,W)`.
    Returns:
        torch.Tensor: RGBA version of the image with shape :math:`(*,4,H,W)`.
    .. note:: The current functionality is NOT supported by Torchscript.
    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_rgba(input, 1.) # 2x4x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    if not isinstance(alpha_val, (float, torch.Tensor)):
        raise TypeError(f"alpha_val type is not a float or torch.Tensor. Got {type(alpha_val)}")

    # add one channel
    r, g, b = torch.chunk(image, image.shape[-3], dim=-3)

    a: torch.Tensor = cast(torch.Tensor, alpha_val)

    if isinstance(alpha_val, float):
        a = torch.full_like(r, fill_value=float(alpha_val))

    return torch.cat([r, g, b, a], dim=-3)


def bgr_to_rgba(image, alpha_val):
    r"""Convert an image from BGR to RGBA.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    if not isinstance(alpha_val, (float, torch.Tensor)):
        raise TypeError(f"alpha_val type is not a float or torch.Tensor. Got {type(alpha_val)}")

    # convert first to RGB, then add alpha channel
    x_rgb: torch.Tensor = bgr_to_rgb(image)
    return rgb_to_rgba(x_rgb, alpha_val)


def rgba_to_rgb(image):
    r"""Convert an image from RGBA to RGB.

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(f"Input size must have a shape of (*, 4, H, W).Got {image.shape}")

    # unpack channels
    r, g, b, a = torch.chunk(image, image.shape[-3], dim=-3)

    # compute new channels
    a_one = torch.tensor(1.) - a
    r_new = a_one * r + a * r
    g_new = a_one * g + a * g
    b_new = a_one * b + a * b

    return torch.cat([r, g, b], dim=-3)


def rgba_to_bgr(image):
    r"""Convert an image from RGBA to BGR.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(f"Input size must have a shape of (*, 4, H, W).Got {image.shape}")

    # convert to RGB first, then to BGR
    x_rgb: torch.Tensor = rgba_to_rgb(image)

    return rgb_to_bgr(x_rgb)


def _extract_device_dtype(tensor_list):
    """Check if all the input are in the same device (only if when they are torch.Tensor).
    If so, it would return a tuple of (device, dtype). Default: (cpu, ``get_default_dtype()``).
    Returns:
        [torch.device, torch.dtype]
    """
    device, dtype = None, None
    for tensor in tensor_list:
        if tensor is not None:
            if not isinstance(tensor, (torch.Tensor,)):
                continue
            _device = tensor.device
            _dtype = tensor.dtype
            if device is None and dtype is None:
                device = _device
                dtype = _dtype
            elif device != _device or dtype != _dtype:
                raise ValueError("Passed values are not in the same device and dtype."
                                 f"Got ({device}, {dtype}) and ({_device}, {_dtype}).")
    if device is None:
        # TODO: update this when having torch.get_default_device()
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()

    return (device, dtype)


def normalize_min_max(x, min_val=0., max_val=1., eps=1e-6):
    r"""Normalise an image tensor by MinMax and re-scales the value between a range.

    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"data should be a tensor. Got: {type(x)}.")

    if not isinstance(min_val, float):
        raise TypeError(f"'min_val' should be a float. Got: {type(min_val)}.")

    if not isinstance(max_val, float):
        raise TypeError(f"'b' should be a float. Got: {type(max_val)}.")

    if len(x.shape) != 4:
        raise ValueError(f"Input shape must be a 4d tensor. Got: {x.shape}.")

    B, C, H, W = x.shape

    x_min = x.view(B, C, -1).min(-1)[0].view(B, C, 1, 1)
    x_max = x.view(B, C, -1).max(-1)[0].view(B, C, 1, 1)

    x_out = ((max_val - min_val) * (x - x_min) / (x_max - x_min + eps) + min_val)

    return x_out.expand_as(x), x_min, x_max


def denormalize_min_max(x, x_min, x_max, eps=1e-6):
    r"""Normalise an image tensor by MinMax and re-scales the value between a range.

    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"data should be a tensor. Got: {type(x)}.")

    if not isinstance(x_min, torch.Tensor):
        raise TypeError(f"data should be a tensor. Got: {type(x)}.")

    if not isinstance(x_max, torch.Tensor):
        raise TypeError(f"data should be a tensor. Got: {type(x)}.")

    if len(x.shape) != 4:
        raise ValueError(f"Input shape must be a 4d tensor. Got: {x.shape}.")

    x_out = (x_max - x_min) * x + x_min

    return x_out


def normal_transform_pixel(height, width, eps=1e-14, device=None, dtype= None):
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    """
    tr_mat = torch.tensor([[1.0, 0.0, -1.0],
                           [0.0, 1.0, -1.0],
                           [0.0, 0.0, 1.0]], device=device, dtype=dtype)  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def normalize_homography(dst_pix_trans_src_pix, dsize_src, dsize_dst):
    r"""Normalize a given homography in pixels to [-1, 1].

    """
    check_is_tensor(dst_pix_trans_src_pix)

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError("Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}"
                         .format(dst_pix_trans_src_pix.shape))

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix = normal_transform_pixel(src_h, src_w,
                                                    device=dst_pix_trans_src_pix.device,
                                                    dtype=dst_pix_trans_src_pix.dtype).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)

    dst_norm_trans_dst_pix = normal_transform_pixel(dst_h, dst_w,
                                                   device=dst_pix_trans_src_pix.device,
                                                   dtype=dst_pix_trans_src_pix.dtype).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm = (dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm))
    return dst_norm_trans_src_norm