from enum import Enum
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF, InterpolationMode

from ..boxes import resize_boxes, clip_boxes

try:
    from PIL import Image
except ImportError:
    Image = None

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

# Pillow >=v9.1.0 use a slightly different naming scheme for filters.
# Set pillow_interp_codes according to the naming scheme used.
if Image is not None:
    if hasattr(Image, 'Resampling'):
        pillow_interp_codes = {
            'nearest': Image.Resampling.NEAREST,
            'bilinear': Image.Resampling.BILINEAR,
            'bicubic': Image.Resampling.BICUBIC,
            'box': Image.Resampling.BOX,
            'lanczos': Image.Resampling.LANCZOS,
            'hamming': Image.Resampling.HAMMING
        }
    else:
        pillow_interp_codes = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'box': Image.BOX,
            'lanczos': Image.LANCZOS,
            'hamming': Image.HAMMING
        }
        
torchvision_interp_codes = {
    'nearst': InterpolationMode.NEAREST,
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'box': InterpolationMode.BOX,
    'hamming': InterpolationMode.HAMMING,
    'lanczos': InterpolationMode.LANCZOS,
}


def get_image_shape(img: ImgType) -> List[int]:
    if isinstance(img, torch.Tensor):
        return list(img.shape[-2:])
    return list(img.shape[:2])


def imresize(img: ImgType,
             size: Optional[Union[int, Tuple[int, int]]] = None,
             scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
             interpolation: str = 'bilinear',
             backend: str = 'cv2',
             antialias: Optional[bool] = True) -> ImgType:
    """Resize image to the given size.

    Args:
        img (ndarray | tensor): The input image.
        size (int | Tuple[int, int] | None): Target size. if size is a tuple 
            like (w, h), output size will be matched to this. If size is an 
            int, bigger edge of the image will bo matched to this number.
            i.e., If height > width, then image will be rescaled to
            (size, size * width / height).
        scale_factor (float | Tuple[float, float] | None): The scaling factor.
            If it is a float number, image will be scaled while keeping the 
            aspect ratio, else it is a tuple like (w_factor, h_factor).
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        backend (str | None): The resize backend type for array image. Options 
            are `cv2`, `pillow`.  Default: `cv2`.
        antialias (bool, optional): The `antialias` arguemnt for 
            :func:torchvision.transfroms.functional.resize`. Only for torch.Tensor image. See more details at 
            `https://github.com/pytorch/vision/blob/b1123cfd543d35d282de9bb28067e48ebec18afe/torchvision/transforms/v2/_geometry.py#L99`
    Returns:
        ndarray | tensor: The resized image.
    """
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported for resize.'
                         f"Supported backends are 'cv2', 'pillow'")
    # get target size.
    h, w = get_image_shape(img)
    if size is not None and scale_factor is not None:
        raise ValueError(
            'only one of `size` or `scale_factor` should be defined') 
    elif size is not None:
        if isinstance(size, int):
            new_w, new_h = int(size * w / h + 0.5), size if w <= h else \
                           size, int(size * h / w + 0.5)
        else:
            new_w, new_h = size
    elif scale_factor is not None:
        if isinstance(scale_factor, (float, int)):
            new_w, new_h = int(w * scale_factor + 0.5), int(scale_factor * h 
                                                            + 0.5)
        else:
            new_w, new_h = int(w * scale_factor[0] + 0.5), int(
                h * scale_factor[1] + 0.5)
    else:
        raise ValueError('either `size` or `scale_factor` should be defined')

    if isinstance(img, np.ndarray):
        if backend == 'pillow':
            assert img.dtype == np.uint8, (f'Pillow backend only support '
                f'uint8 type, but get {img.dtype}')
            pil_image = Image.fromarray(img)
            pil_image = pil_image.resize(
                (new_w, new_h), pillow_interp_codes[interpolation])
            return np.array(pil_image)
        else:
            return cv2.resize(img, (new_w, new_h), 
                              interpolation=cv2_interp_codes[interpolation])
    else:
        return TF.resize(img, (new_h, new_w), 
                         torchvision_interp_codes[interpolation], 
                         antialias=antialias)


class PositionType(Enum):
    """Enumerates the types of padding positions.

    Attributes
        AROUND (str): Specifies that the padding should be placed around the
            image. i.e. the image should be palced at the center.
        TOP_LEFT (str): Specifies that the padding should be placed at the 
            top-left.
        TOP_RIGHT (str): Specifies that the padding should be placed at the 
            top-right.
        BOTTOM_LEFT (str): Specifies that the padding should be placed at the 
            bottom-left.
        BOTTOM_RIGHT (str): Specifies that the padding should be placed at the 
            bottom-right.
    """

    AROUND = "around"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


def _parse_pad_padding(padding: Optional[List[int]] = None, 
                       ori_size: Optional[Tuple[int, int]] = None,
                       new_size: Optional[Tuple[int, int]] = None,
                       pad_position: PositionType = PositionType.AROUND
) -> List[int]:
    if padding is not None:
        if len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        else:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]
    else:
        h, w = ori_size
        new_h, new_w = new_size
        if pad_position == PositionType.AROUND:
            pad_left = max(new_w - w, 0) // 2
            pad_right = max(new_w - pad_left - w, 0)
            pad_top = max(new_h - h, 0) // 2
            pad_bottom = max(new_h - pad_top - h, 0)
        elif pad_position == PositionType.TOP_LEFT:
            pad_right, pad_bottom = 0, 0
            pad_left = max(new_w - w, 0)
            pad_top = max(new_h - h, 0)
        elif pad_position == PositionType.TOP_RIGHT:
            pad_left, pad_bottom = 0, 0
            pad_right = max(new_w - w, 0)
            pad_top = max(new_h - h, 0)
        elif pad_position == PositionType.BOTTOM_LEFT:
            pad_right, pad_top = 0, 0
            pad_left = max(new_w - w, 0)
            pad_bottom = max(new_h - h, 0)
        elif pad_position == PositionType.BOTTOM_RIGHT:
            pad_left, pad_top = 0, 0
            pad_right = max(new_w - w, 0)
            pad_bottom = max(new_h - h, 0)

    return [pad_left, pad_top, pad_right, pad_bottom]
    

def impad(img: ImgType,
          padding: Optional[Union[int, Sequence[int]]] = None,
          min_size: Optional[Union[int, Tuple[int, int]]] = None,
          size_divisor: Optional[int] = None,
          to_square: bool = False,
          pad_position: Union[PositionType, str] = PositionType.AROUND,
          pad_val: Union[Number, Sequence[Number]] = 0,
          padding_mode: str = 'constant',
          return_padding: bool = False):
    """Pad the given image on all sides or to a certain shape with specified 
    padding mode and padding value.

    Args:
        img (ndarray | tensor): Image to be padded.
        padding (int or Sequence[int]): Padding on each border. If a single int 
            is provided this is used to pad all borders. If tuple of length 2 
            is provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `padding` can not be set together with 
            `min_size`, `size_divisor` or `to_square`. And `pad_position` has 
            no effect on `padding`.
        min_size(int | tuple[int, int]): Fixed minimal padding size. When 
            `to_square` is set to `True`, min_size is an int number. 
            Conversely, min_size must be a tuple representing the expected 
            padding shape (h, w).  Default: None.
        size_divisor (int, optional): The divisor of padded size. Defaults to
            None. Note that `min_size` and `size_divisor` can not be both set.
        to_square (bool): Whether to pad the image into a square. 
            Defaults to False.
        pad_position: (str | PositionType): Position of the padding.
            Should be PositionType.AROUND or PositionType.TOP_LEFT or 
            PositionType.TOP_RIGHT or PositionType.BOTTOM_LEFT or 
            PositionType.BOTTOM_RIGHT. Default: PositionType.AROUND.
            Note that this arguemet has no effect when `padding` is gaven.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]
        return_padding: Whether to return padding.

    Returns:
        tuple | ndarray: (padding, padded_image) or padded_image
    """
    assert (min_size is None) or (size_divisor is None), \
        'only one of `size` and `size_divisor` can be set'
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
    pad_position = PositionType(pad_position)

    h, w = get_image_shape(img)
    if padding is not None:
        if (min_size is not None) or (size_divisor is not None) or to_square:
            raise ValueError('`padding` can not be set together with '
                             '`min_size`, `size_divisor` or `to_square`')
        if isinstance(padding, (list, tuple)):
            if len(padding) not in [ 2, 4]:
                raise ValueError(
                    f"Padding must be an int or a 2, or 4 element tuple, "
                    f"not a {len(padding)} element tuple")
        elif isinstance(padding, int):
            padding = [padding]
        else:
            raise ValueError(
                 f"Padding must be an int or a 1, 2, or 4 element tuple, "
                 f"bug get `padding={padding}`")
        new_h, new_w = None, None
    elif to_square: 
        if min_size is None:
            long_edge = max(h, w)
        elif isinstance(min_size, int):
            long_edge = max(h, w, min_size)  
        else:
            raise ValueError(f"`min_size` must be None or an int number when "
                             f"`to_square=True`, but get {min_size}")
        if size_divisor is not None:
            long_edge = ((long_edge + size_divisor - 1) // size_divisor
                         ) * size_divisor
        new_h, new_w = long_edge, long_edge
    elif min_size is not None:
        if isinstance(min_size, (tuple, list)) and isinstance(min_size[0], 
                                                                int):
            new_h, new_w = min_size
        else:
            raise ValueError(f"`min_size` must a 2 int sequence, when "
                             f"`to_square=False`, but get {min_size}")
    elif size_divisor is not None:
        if not isinstance(size_divisor, int):
            raise TypeError(f"`size_divisor` must be an int number, "
                             f"but get a {type(size_divisor)}")
        new_h = ((h + size_divisor - 1) // size_divisor) * size_divisor
        new_w = ((w + size_divisor - 1) // size_divisor) * size_divisor
    else:
        raise ValueError(f"None of `padding`, `min_size`, `size_divisor` and "
                         f"`to_square` has been set")
        
    padding = _parse_pad_padding(padding, (h, w), (new_h, new_w), pad_position) 
       
    if isinstance(img, np.ndarray):
         # check pad_val
        if isinstance(pad_val, (tuple, list)):
            if img.ndim == 3:
                assert len(pad_val) == img.shape[-1]
            elif img.ndim == 2:
                assert len(pad_val) == 1
        elif isinstance(pad_val, Number):
            assert img.ndim == 2 or img.shape[-1] == 1
        else:
            raise TypeError(f'pad_val must be a int or a tuple. when pad with'
                            f'array image. But received {type(pad_val)}')
        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            cv2_border_modes[padding_mode],
            value=pad_val)
    else:
        if not isinstance(pad_val, Number):
            raise TypeError(f'pad_val must be a int or a tuple. when pad with'
                            f'tensor image. But received {type(pad_val)}')
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

    def _ndarray_patch_fill(patch, img, p_x1, p_y1, i_x1, i_y1, w, h):
        patch[p_y1: p_y1+h, p_x1: p_x1+w, ...] = img[
            i_y1: i_y1+h, i_x1: i_x1+w, ...]
    
    def _tensor_patch_fill(patch, img, p_x1, p_y1, i_x1, i_y1, w, h):
        patch[..., p_y1: p_y1+h, p_x1: p_x1+w] = img[
            ..., i_y1: i_y1+h, i_x1: i_x1+w]

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
        x1, y1, x2, y2 = clipped_boxes[i, :]
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
            patch = patch_fill(patch, img, x_start, y_start, x1, y1, w, h)
        patches.append(patch)

    returned_boxes = clipped_boxes if pad_fill is None else scaled_boxes
    if bboxes.ndim == 1:
        return patches[0], returned_boxes[0]
    else:
        return patches, returned_boxes