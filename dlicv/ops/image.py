from numbers import Number
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF, InterpolationMode

from .boxes import resize_boxes, clip_boxes

ImgType = Union[np.ndarray, torch.Tensor]

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

cv2_border_modes = {
    'constant': cv2.BORDER_CONSTANT,
    'replicate': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT,
    'wrap': cv2.BORDER_WRAP,
    'reflect_101': cv2.BORDER_REFLECT_101,
    'transparent': cv2.BORDER_TRANSPARENT,
    'isolated': cv2.BORDER_ISOLATED
}

torchvision_interp_codes = {
    'nearst': InterpolationMode.NEAREST,
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'box': InterpolationMode.BOX,
    'hamming': InterpolationMode.HAMMING,
    'lanczos': InterpolationMode.LANCZOS,
}


def _get_keep_ratio_size(target_size: Tuple[int, int], 
                         old_size: Tuple[int, int]) -> tuple:
    h, w = old_size
    scale_factor = min(target_size[0] / h, target_size[1] / w)
    new_w = int(w * scale_factor + 0.5)
    new_h = int(h * scale_factor + 0.5)
    return new_h, new_w

    
def imresize(img: ImgType,
             size: Tuple[int, int],
             keep_ratio = False,
             interpolation: str = 'bilinear') -> ImgType:
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
        if keep_ratio:
            size = _get_keep_ratio_size(size, (h, w))
        
        return cv2.resize(img, size, 
                          interpolation=cv2_interp_codes[interpolation])
    else:
        h, w = img.shape[-2:]
        if keep_ratio:
            size = _get_keep_ratio_size(size, (h, w))
        return TF.resize(img, size, torchvision_interp_codes[interpolation], 
                         antialias=True)

    
def impad(img: ImgType,
          size: Optional[Tuple[int, int]] = None,
          size_divisor: Optional[int] = None,
          pad_val: Number = 0,
          mode: str = 'around',
          to_square: bool = False,
          padding_mode: str = 'constant',
          return_padding: bool = False):
    if to_square:
        assert size is None, \
                'The size must be None when pad2square is True'
    else:
        assert (size is None) ^ (size_divisor is None), \
            'only one of size and size_divisor should be valid'
    assert mode in ['around', 'rb']
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    def _get_padding(h, w) -> tuple:
        target_size = None
        if to_square:
            max_size = max(h, w)
            target_size = (max_size, max_size)
        if size_divisor is not None:
            if target_size is None:
                target_size = (h, w)
            pad_h = int(
                np.ceil(target_size[0] / size_divisor)) * size_divisor
            pad_w = int(
                np.ceil(target_size[1] / size_divisor)) * size_divisor
            target_size = (pad_h, pad_w)
        elif size is not None:
            target_size = size
        if mode == 'around':
            pad_top = max(target_size[0] - h, 0) // 2
            pad_bottom = max(target_size[0] - pad_top - h, 0)
            pad_left = max(target_size[1] - w, 0) // 2
            pad_right = max(target_size[1] - pad_left - w, 0)
        else:
            pad_top, pad_left = 0, 0
            pad_right = max(target_size[1] - w, 0)
            pad_bottom = max(target_size[0] - h, 0)
        return pad_left, pad_top, pad_right, pad_bottom
    
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
        pad_val = tuple(pad_val for _ in range(img.shape[2]))
        padding = _get_padding(h, w)
        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            cv2_border_modes[padding_mode],
            value = pad_val)
    else:
        h, w = img.shape[-2:]
        padding = _get_padding(h, w)
        img = TF.pad(img, padding, pad_val, padding_mode)
    if return_padding:
        return img, padding
    else:
        return img

    
def imrotate(img: ImgType,
             angle: float,
             center: Optional[Tuple[float, float]] = None,
             border_value: Union[Number, List[Number]] = 0,
             interpolation: str = 'bilinear',
             auto_bound: bool = False) -> ImgType:
    """Rotate an image.

    Args:
        img (np.ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value used in case of a constant border.
            Defaults to 0.
        interpolation (str): Same as :func:`resize`.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.
        border_mode (str): Pixel extrapolation method. Defaults to 'constant'.

    Returns:
        np.ndarray: The rotated image.
    """
    if isinstance(img, np.ndarray):
        if center is not None and auto_bound:
            raise ValueError('`auto_bound` conflicts with `center` '
                             'when rotate ndarray img')
        h, w = img.shape[:2]
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
        rotated = cv2.warpAffine(
            img,
            matrix, (w, h),
            flags=cv2_interp_codes[interpolation],
            borderValue=border_value)
    else:
        rotated = TF.rotate(img, 
                            -angle,
                            torchvision_interp_codes[interpolation],
                            auto_bound,
                            center,
                            border_value)
    return rotated


def imcrop(img: ImgType,
           bboxes: Union[torch.Tensor, np.ndarray],
           scale: float = 1.0,
           pad: float = 0,
           pad_fill: Union[float, list, None] = None,
) -> Union[ImgType, List[ImgType]]:
    def _ndarray_img_slice(img, x1, y1, x2, y2):
        return img[y1:y2 + 1, x1:x2 + 1, ...]

    def _tensor_img_slice(img, x1, y1, x2, y2):
        return img[..., y1:y2 + 1, x1:x2 + 1]

    def _ndarray_patch_fill(patch, img, x1, y1, w, h):
        patch[y1: y1+h, x1: x1+w, ...] = img[y1: y1+h, x1: x1+w, ...]
    
    def _tensor_patch_fill(patch, img, x1, y1, w, h):
        patch[..., y1: y1+h, x1: x1+w] = img[..., y1: y1+h, x1: x1+w]

    def _create_ndarray_patch(pad_fill, img, patch_h, patch_w, chn):
        if chn == 1:
            patch_shape = (patch_h, patch_w)
        else:
            patch_shape = (patch_h, patch_w, chn)
        patch = np.array(pad_fill, dtype=img.dtype) * np.ones(
            patch_shape, dtype=img.dtype)
        return patch

    def _create_tensor_patch(pad_fill, img, patch_h, patch_w, chn):
        patch_shape = (chn, patch_h, patch_w)
        patch = img.new_tensor(pad_fill)[..., None, None] * img.new_ones(
            patch_shape)
        return patch

    assert pad >= 0
    _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
        patch_fill = _ndarray_patch_fill
        create_patch = _create_ndarray_patch
        img_slice = _ndarray_img_slice
        chn = 1 if img.ndim == 2 else img.shape[2]
        if scale == 1.0:
            scaled_boxes = _bboxes.copy()
        else:
            scaled_boxes = resize_boxes(_bboxes, (scale, scale))
            scaled_boxes = scaled_boxes.round().astype(np.int32)
        if pad > 0:
            scaled_boxes += (-pad, -pad, pad, pad)
    else:
        h, w = img.shape[-2:]
        patch_fill = _tensor_patch_fill
        create_patch = _create_tensor_patch
        img_slice = _tensor_img_slice
        chn = img.shape[-3]
        if scale == 1.0:
            scaled_boxes = _bboxes.clone()
        else:
            scaled_boxes = resize_boxes(_bboxes, (scale, scale))
            scaled_boxes = scaled_boxes.round().to(torch.int32)
        if pad > 0:
            scaled_boxes += scaled_boxes.new_tensor((-pad, -pad, pad, pad))
    if pad_fill is not None:
        if isinstance(pad_fill, (int, float)):
            pad_fill = [pad_fill for _ in range(chn)]
        assert len(pad_fill) == chn
    
    patches = []
    clipped_boxes = clip_boxes(scaled_boxes, (h - 1, w - 1))
    for i in range(clipped_boxes.shape[0]):
        x1, y1, x2, y2 = scaled_boxes[i, :]
        if pad_fill is None:
            patch = img_slice(img, x1, y1, x2, y2)
        else:
            _x1, _y1, _x2, _y2 = scaled_boxes[i, :]
            patch_h = _y2 - _y1 + 1
            patch_w = _x2 - _x1 + 1
            patch = create_patch(pad_fill, img, patch_h, patch_w, chn)
            x_start = 0 if _x1 >= 0 else -_x1
            y_start = 0 if _y1 >= 0 else -_y1
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            patch = patch_fill(patch, img, x_start, y_start, w, h)
        patches.append(patch)

    if bboxes.ndim == 1:
        return patches[0], clipped_boxes[0]
    else:
        return patches, clipped_boxes