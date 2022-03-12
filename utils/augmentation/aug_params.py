
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from .misc import _adapted_uniform

def _adapted_sampling(p, shape, device, dtype):
    r"""The uniform sampling function that accepts 'same_on_batch'.
    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.
    """
    _bernoulli = Bernoulli(torch.tensor(float(p), device=device, dtype=dtype))
    target = _bernoulli.sample((shape,)).bool()
    return target

# --------------------------------------
#             Geometric
# --------------------------------------

def RandomHorizontalFlip_params(p, batch_size, img_size, device, dtype):

    target = _adapted_sampling(p, batch_size, device, dtype)
    # params
    params = dict()
    params["target"] = target
    return params


def RandomVerticalFlip_params(p, batch_size, img_size, device, dtype):
    target = _adapted_sampling(p, batch_size, device, dtype)
    # params
    params = dict()
    params["target"] = target
    return params

def RandomPerspective_params(p, distortion_scale, batch_size, img_size, device, dtype):
    target = _adapted_sampling(p, batch_size, device, dtype)

    height, width = img_size

    distortion_scale = torch.as_tensor(distortion_scale, device=device, dtype=dtype)

    assert distortion_scale.dim() == 0 and 0 <= distortion_scale <= 1, \
        f"'distortion_scale' must be a scalar within [0, 1]. Got {distortion_scale}."

    assert type(height) == int and height > 0 and type(width) == int and width > 0, \
        f"'height' and 'width' must be integers. Got {height}, {width}."

    start_points = torch.tensor([[
        [0., 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ]], device=device, dtype=dtype).expand(batch_size, -1, -1)

    # generate random offset not larger than half of the image
    fx = distortion_scale * width / 2
    fy = distortion_scale * height / 2

    factor = torch.stack([fx, fy], dim=0).view(-1, 1, 2)

    pts_norm = torch.tensor([[
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1]
    ]], device=device, dtype=dtype)

    rand_val = _adapted_uniform(start_points.shape,
                                torch.tensor(0, device=device, dtype=dtype),
                                torch.tensor(1, device=device, dtype=dtype)
                                ).to(device=device, dtype=dtype)

    end_points = start_points + factor * rand_val * pts_norm

    params = dict()
    params["target"] = target
    params["start_points"] = start_points
    params["end_points"] = end_points

    return params

def RandomAffine_params(p, theta, h_trans, v_trans, scale, shear, batch_size, img_size, device, dtype):

    theta = [-theta, theta]
    translate = [h_trans, v_trans]
    scale = scale
    shear = shear

    interpolation = 'bilinear'
    padding_mode = 'zeros'
    align_corners = False

    height, width = img_size

    target = _adapted_sampling(p, batch_size, device, dtype)

    assert isinstance(width, (int,)) and isinstance(height, (int,)) and width > 0 and height > 0, \
        f"`width` and `height` must be positive integers. Got {width}, {height}."

    degrees = torch.as_tensor(theta).to(device=device, dtype=dtype)
    angle = _adapted_uniform((batch_size,), degrees[0], degrees[1]).to(device=device, dtype=dtype)

    # compute tensor ranges
    if scale is not None:
        scale = torch.as_tensor(scale).to(device=device, dtype=dtype)

        assert len(scale.shape) == 1 and (len(scale) == 2), \
            f"`scale` shall have 2 or 4 elements. Got {scale}."

        _scale = _adapted_uniform((batch_size,), scale[0], scale[1]).unsqueeze(1).repeat(1, 2)
    else:
        _scale = torch.ones((batch_size, 2), device=device, dtype=dtype)

    if translate is not None:
        translate = torch.as_tensor(translate).to(device=device, dtype=dtype)

        max_dx = translate[0] * width
        max_dy = translate[1] * height

        translations = torch.stack([
            _adapted_uniform((batch_size,), max_dx * 0, max_dx),
            _adapted_uniform((batch_size,), max_dy * 0, max_dy)
        ], dim=-1).to(device=device, dtype=dtype)

    else:
        translations = torch.zeros((batch_size, 2), device=device, dtype=dtype)

    center = torch.tensor([width, height], device=device, dtype=dtype).view(1, 2) / 2. - 0.5
    center = center.expand(batch_size, -1)

    if shear is not None:
        shear = torch.as_tensor(shear).to(device=device, dtype=dtype)

        sx = _adapted_uniform((batch_size,), shear[0], shear[1]).to(device=device, dtype=dtype)
        sy = _adapted_uniform((batch_size,), shear[0], shear[1]).to(device=device, dtype=dtype)

        sx = sx.to(device=device, dtype=dtype)
        sy = sy.to(device=device, dtype=dtype)
    else:
        sx = sy = torch.tensor([0] * batch_size, device=device, dtype=dtype)

    # params
    params = dict()
    params["target"] = target
    params["translations"] = translations
    params["center"] = center
    params["scale"] = _scale
    params["angle"] = angle
    params["sx"] = sx
    params["sy"] = sy
    return params


def RandomRotation_params(p, theta, batch_size, img_size, device, dtype):
    r"""Get parameters for ``rotate`` for a random rotate transform.

            """
    theta = [-theta, theta]
    interpolation = 'bilinear'
    padding_mode = 'zeros'
    align_corners = False
    target = _adapted_sampling(p, batch_size, device, dtype)

    angle = torch.as_tensor(theta).to(device=device, dtype=dtype)
    angle = _adapted_uniform((batch_size,), angle[0], angle[1]).to(device=device, dtype=dtype)

    # params
    params = dict()
    params["target"] = target
    params["angle"] = angle

    return params

# --------------------------------------
#             Photometric
# --------------------------------------

def ColorJitter_params(p, brightness, contrast, saturation, hue, batch_size, img_size, device, dtype):
    target = _adapted_sampling(p, batch_size, device, dtype)

    if brightness is not None:
        brightness = torch.as_tensor(brightness).to(device=device, dtype=dtype)
    else:
        brightness = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

    if contrast is not None:
        contrast = torch.as_tensor(contrast).to(device=device, dtype=dtype)
    else:
        contrast = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

    if saturation is not None:
        saturation = torch.as_tensor(saturation).to(device=device, dtype=dtype)
    else:
        saturation = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

    if hue is not None:
        hue = torch.as_tensor(hue).to(device=device, dtype=dtype)
    else:
        hue = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

    brightness_factor = _adapted_uniform((batch_size,), brightness[0], brightness[1]).to(device=device, dtype=dtype)
    contrast_factor = _adapted_uniform((batch_size,), contrast[0], contrast[1]).to(device=device, dtype=dtype)
    saturation_factor = _adapted_uniform((batch_size,), saturation[0], saturation[1]).to(device=device, dtype=dtype)
    hue_factor = _adapted_uniform((batch_size,), hue[0], hue[1]).to(device=device, dtype=dtype)

    # Params
    params = dict()

    params["brightness_factor"] = brightness_factor
    params["contrast_factor"] = contrast_factor
    params["hue_factor"] = hue_factor
    params["saturation_factor"] = saturation_factor

    params["order"] = torch.randperm(4, device=device, dtype=dtype).long()
    params["target"] = target

    return params

def RandomSolarize_params(p, thresholds, additions, batch_size, img_size, device, dtype):
    target = _adapted_sampling(p, batch_size, device, dtype)

    if thresholds is not None:
        thresholds = torch.as_tensor(thresholds).to(device=device, dtype=dtype)
    else:
        thresholds = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

    if additions is not None:
        additions = torch.as_tensor(additions).to(device=device, dtype=dtype)
    else:
        additions = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

    thresholds = _adapted_uniform((batch_size,), thresholds[0], thresholds[1]).to(device=device, dtype=dtype)
    additions = _adapted_uniform((batch_size,), additions[0], additions[1]).to(device=device, dtype=dtype)

    # Params
    params = dict()

    params["thresholds"] = thresholds
    params["additions"] = additions

    params["target"] = target

    return params

def RandomPosterize_params(p, bits, batch_size, img_size, device, dtype):
    target = _adapted_sampling(p, batch_size, device, dtype)

    if bits is not None:
        bits = torch.as_tensor(bits).to(device=device, dtype=dtype)
    else:
        bits = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

    bits = _adapted_uniform((batch_size,), bits[0], bits[1]).to(device=device, dtype=dtype).int()

    # Params
    params = dict()

    params["bits"] = bits
    params["target"] = target
    return params

def RandomSharpness_params(p, sharpness, batch_size, img_size, device, dtype):
    target = _adapted_sampling(p, batch_size, device, dtype)

    if sharpness is not None:
        sharpness = torch.as_tensor(sharpness).to(device=device, dtype=dtype)
    else:
        sharpness = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

    sharpness = _adapted_uniform((batch_size,), sharpness[0], sharpness[1]).to(device=device, dtype=dtype)

    # Params
    params = dict()

    params["sharpness"] = sharpness
    params["target"] = target

    return params

def RandomEqualize_params(p, batch_size, img_size, device, dtype):
    target = _adapted_sampling(p, batch_size, device, dtype)

    # Params
    params = dict()

    params["target"] = target
    return params

def RandomGrayscale_params(p, batch_size, img_size, device, dtype):
    target = _adapted_sampling(p, batch_size, device, dtype)

    # Params
    params = dict()

    params["target"] = target
    return params

# --------------------------------------
#             Filters
# --------------------------------------

def RandomMotionBlur_params(p, kernel_size, angle, direction, batch_size, img_size, device, dtype,
                            interpolation='bilinear', border_mode='zeros', align_corners=False):

    target = _adapted_sampling(p, batch_size, device, dtype)

    if kernel_size is not None:
        kernel_size = torch.as_tensor(kernel_size).to(device=device, dtype=dtype)
    else:
        kernel_size = torch.as_tensor([3, 3]).to(device=device, dtype=dtype)

    if angle is not None:
        angle = torch.as_tensor(angle).to(device=device, dtype=dtype)
    else:
        angle = torch.as_tensor([0, 0]).to(device=device, dtype=dtype)

    if direction is not None:
        direction = torch.as_tensor(direction).to(device=device, dtype=dtype)
    else:
        direction = torch.as_tensor([0, 0]).to(device=device, dtype=dtype)

    kernel_size = _adapted_uniform((1,),
                                   kernel_size[0] // 2,
                                   kernel_size[1] // 2).to(device=device, dtype=dtype).int() * 2 + 1

    angle = _adapted_uniform((batch_size,), angle[0], angle[1]).to(device=device, dtype=dtype)
    direction = _adapted_uniform((batch_size,), direction[0], direction[1]).to(device=device, dtype=dtype)

    # Params
    params = dict()

    params["kernel_size"] = kernel_size
    params["angle"] = angle
    params["direction"] = direction

    params["target"] = target

    return params
