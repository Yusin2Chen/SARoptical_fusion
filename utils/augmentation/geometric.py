import torch
import torch.nn.functional as F

from .misc import (
    pi,
    deg2rad,
    rad2deg,
    bbox_generator,
    normalize_homography,
    _compute_tensor_center,
    _compute_rotation_matrix,
    check_is_tensor,
    create_meshgrid,
    get_rotation_matrix2d,
    get_shear_matrix2d,
    convert_affinematrix_to_homography,
    convert_points_from_homogeneous,
    convert_points_to_homogeneous,
)

# --------------------------------------
#             Centerlize
# --------------------------------------


def centralize(inp, valid_size):
    r"""Applies transformation to a tensor image to center image FOE.

    The transformation is computed so that the image center is kept invariant.
    """
    # note input valid size is different than shape
    valid_size = [torch.tensor([item[0], item[1]]) for item in valid_size]

    valid_size = torch.stack(valid_size).to(device=inp.device, dtype=inp.dtype)

    max_size = inp.shape[-2:]
    max_size = torch.tensor([max_size[0], max_size[1]]).float().to(device=inp.device, dtype=inp.dtype)

    coef_size = max_size/2 - valid_size/2

    translation = [torch.tensor([1, 0, coef[1], 0, 1, coef[0]]).view(2, 3) for coef in coef_size]
    translation = torch.stack(translation, dim=0)

    # pad transform to get Bx3x3
    transform = convert_affinematrix_to_homography(translation).to(device=inp.device, dtype=inp.dtype)

    size = inp.shape[-2:]
    inp = warp_affine(inp, transform, size)

    bbx = [torch.tensor([coef[0], coef[1], size[0], size[1]]) for coef, size in zip(coef_size, valid_size)]
    bbx = torch.stack(bbx).to(device=inp.device, dtype=inp.dtype)

    return inp, bbx


# --------------------------------------
#             Flipping
# --------------------------------------

def hflip(tensor):
    """
        Horizontal Flipping
    """

    check_is_tensor(tensor)

    w = tensor.shape[-1]
    return tensor[..., torch.arange(w - 1, -1, -1, device=tensor.device)]


def vflip(tensor):
    """
        Vertical Flipping
    """
    check_is_tensor(tensor)

    h = tensor.shape[-2]

    return tensor[..., torch.arange(h - 1, -1, -1, device=tensor.device), :]

# --------------------------------------
#             Rotations
# --------------------------------------


def rotate(tensor, angle, center=None, interpolation='bilinear', align_corners=False):
    """
        Rotate input Tensor anti-clockwise about its centre.
    """
    check_is_tensor(tensor)
    check_is_tensor(angle)

    if center is not None and not torch.is_tensor(center):
        raise TypeError("Input center type is not a torch.Tensor. Got {}"
                        .format(type(center)))
    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the rotation center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix

    angle = angle.expand(tensor.shape[0])
    center = center.expand(tensor.shape[0], -1)

    rotation_matrix = _compute_rotation_matrix(angle, center)
    # warp using the affine transform
    return affine(tensor, rotation_matrix[..., :2, :3],
                  interpolation=interpolation,
                  align_corners=align_corners)

# --------------------------------------
#             Croping
# --------------------------------------


# --------------------------------------
#             Affine
# --------------------------------------
def get_affine_matrix2d(translations, center, scale, angle, sx=None, sy=None):
    r"""Composes affine matrix from the components.
    """
    # scaled rotation
    transform = get_rotation_matrix2d(center, -angle, scale)

    # translations
    transform[..., 2] += translations  # tx/ty

    # pad transform to get Bx3x3
    transform_h = convert_affinematrix_to_homography(transform)
    if any([s is not None for s in [sx, sy]]):
        shear_mat = get_shear_matrix2d(center, sx, sy)
        transform_h = transform_h @ shear_mat
    return transform_h


def warp_affine(src, M, dsize, interpolation='bilinear', padding_mode='zeros', align_corners=False):
    """
        Apply affine transformation to a tensor.
    """
    check_is_tensor(src)
    check_is_tensor(M)

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}"
                         .format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}"
                         .format(M.shape))

    B, C, H, W = src.size()
    dsize_src = (H, W)
    out_size = dsize

    dst_norm_trans_src_norm = normalize_homography(M, dsize_src, out_size)
    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)

    grid = F.affine_grid(src_norm_trans_dst_norm[:, :2, :],
                         [B, C, out_size[0], out_size[1]],
                         align_corners=align_corners)

    return F.grid_sample(src,
                         grid,
                         align_corners=align_corners,
                         mode=interpolation,
                         padding_mode=padding_mode)


def affine(tensor, matrix, interpolation='bilinear', align_corners=False):
    """
        Apply affine transformation to a tensor.
    """
    # warping needs data in the shape of BCHW
    is_unbatched = tensor.ndimension() == 3

    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    matrix = matrix.expand(tensor.shape[0], -1, -1)

    matrix = convert_affinematrix_to_homography(matrix)

    # warp the input tensor
    height = tensor.shape[-2]
    width = tensor.shape[-1]

    warped = warp_affine(tensor, matrix, (height, width),
                         interpolation=interpolation,
                         align_corners=align_corners)

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped,
                               dim=0)

    return warped


# --------------------------------------
#             perspective
# --------------------------------------
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
    points_0_h = torch.bmm(points_1_h,
                           trans_01.permute(0, 2, 1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
    # reshape to the input shape
    shape_inp[-2] = points_0.shape[-2]
    shape_inp[-1] = points_0.shape[-1]
    points_0 = points_0.reshape(shape_inp)
    return points_0


def warp_grid(grid, src_homo_dst):
    r"""Compute the grid to warp the coordinates grid by the homography/ies.
    """
    batch_size: int = src_homo_dst.size(0)
    _, height, width, _ = grid.size()

    # expand grid to match the input batch size
    grid = grid.expand(batch_size, -1, -1, -1)  # NxHxWx2

    if len(src_homo_dst.shape) == 3:  # local homography case
        src_homo_dst = src_homo_dst.view(batch_size, 1, 3, 3)  # Nx1x3x3

    # perform the actual grid transformation,
    # the grid is copied to input device and casted to the same type

    flow = transform_points(src_homo_dst, grid.to(src_homo_dst))  # NxHxWx2

    return flow.view(batch_size, height, width, 2)  # NxHxWx2


def _build_perspective_param(p, q, axis='x'):

    ones = torch.ones_like(p)[..., 0:1]
    zeros = torch.zeros_like(p)[..., 0:1]

    if axis == 'x':
        return torch.cat(
            [
                p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
                -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]
            ], dim=1)

    if axis == 'y':
        return torch.cat(
            [
                zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones,
                -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]

            ], dim=1)

    raise NotImplementedError(f"perspective params for axis `{axis}` is not implemented.")


def get_perspective_transform(src, dst):

    if not isinstance(src, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(src)))
    if not isinstance(dst, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(dst)))
    if not src.shape[-2:] == (4, 2):
        raise ValueError("Inputs must be a Bx4x2 tensor. Got {}"
                         .format(src.shape))
    if not src.shape == dst.shape:
        raise ValueError("Inputs must have the same shape. Got {}"
                         .format(dst.shape))
    if not (src.shape[0] == dst.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Expect {} but got {}"
                         .format(src.shape, dst.shape))

    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    p = []
    for i in [0, 1, 2, 3]:
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'x'))
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'y'))

    # A is Bx8x8
    A = torch.stack(p, dim=1)

    # b is a Bx8x1
    b = torch.stack([
        dst[:, 0:1, 0], dst[:, 0:1, 1],
        dst[:, 1:2, 0], dst[:, 1:2, 1],
        dst[:, 2:3, 0], dst[:, 2:3, 1],
        dst[:, 3:4, 0], dst[:, 3:4, 1],
    ], dim=1)

    # solve the system Ax = b
    X, LU = torch.solve(b, A)

    # create variable to return
    batch_size = src.shape[0]
    M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
    M[..., :8] = torch.squeeze(X, dim=-1)

    return M.view(-1, 3, 3)  # Bx3x3


def homography_warp(patch_src, src_homo_dst, dsize, mode='bilinear', padding_mode='zeros', align_corners=False, normalized_coordinates=True):
    r"""Warp image patchs or tensors by normalized 2D homographies.
    """
    if not src_homo_dst.device == patch_src.device:
        raise TypeError("Patch and homography must be on the same device. \
                         Got patch.device: {} src_H_dst.device: {}.".format(patch_src.device, src_homo_dst.device))

    height, width = dsize
    grid = create_meshgrid(height, width, normalized_coordinates=normalized_coordinates)
    warped_grid = warp_grid(grid, src_homo_dst)

    return F.grid_sample(patch_src, warped_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def transform_warp_impl(src, dst_pix_trans_src_pix, dsize_src, dsize_dst, grid_mode, padding_mode, align_corners):
    """Compute the transform in normalized coordinates and perform the warping.
    """
    dst_norm_trans_src_norm = normalize_homography(dst_pix_trans_src_pix, dsize_src, dsize_dst)

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    return homography_warp(src, src_norm_trans_dst_norm, dsize_dst, grid_mode, padding_mode)


def compute_perspective_transformation(input, params):
    r"""Compute the applied transformation matrix :math: `(*, 3, 3)`.
    """
    perspective_transform = get_perspective_transform(params['start_points'], params['end_points']).type_as(input)

    transform = perspective_transform

    return transform


def warp_perspective(src, M, dsize, interpolation='bilinear', border_mode='zeros', align_corners=False):
    r"""Applies a perspective transformation to an image.
    The function warp_perspective transforms the source image using
    the specified matrix:

    """

    check_is_tensor(src)
    check_is_tensor(M)

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}"
                         .format(src.shape))

    if not (len(M.shape) == 3 and M.shape[-2:] == (3, 3)):
        raise ValueError("Input M must be a Bx3x3 tensor. Got {}"
                         .format(M.shape))

    # launches the warper
    h, w = src.shape[-2:]

    return transform_warp_impl(src, M, (h, w), dsize, interpolation, border_mode, align_corners)


# --------------------------------------
#             Other or not used yet
# --------------------------------------



