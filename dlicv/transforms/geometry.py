from numbers import Number
from typing import Optional, Sequence, Tuple, Union

import dlicv
from dlicv.ops.image.geometric import PositionType
from .base import BaseTransform


class Resize(BaseTransform):
    """Resize images.

    This transform resizes the input image according to ``size`` or 
    ``scale_factor``. 

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Added Keys:

    - scale_factor

    Args:
        img (ndarray | tensor): The input image.
        size (int | Tuple[int, int] | None): Target size. if size is a tuple 
            like (w, h), output size will be matched to this. If size is an 
            int, bigger edge of the image will be matched to this number.
            i.e., If height > width, then image will be rescaled to
            (size, size * width / height).
        scale_factor (float | Tuple[float, float] | None): The scaling factor.
            If it is a float number, image will be scaled while keeping the 
            aspect ratio, else it is a tuple like (w_factor, h_factor). Note 
            that `size` and `scale_factor` can not be both set.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        backend (str | None): The resize backend type for array image. Options 
            are `cv2`, `pillow`. Default: `cv2`.
        antialias (bool, optional): The `antialias` arguemnt for 
            :func:torchvision.transfroms.functional.resize`. Only valid for 
            torch.Tensor image. See more details at 
            `https://github.com/pytorch/vision/blob/b1123cfd543d35d282de9bb28067e48ebec18afe/torchvision/transforms/v2/_geometry.py#L99`
    """
    def __init__(self, 
                 size: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, 
                                              Tuple[float, float]]] = None,
                 interpolation: str = 'bilinear',
                 backend: str = 'cv2',
                 antialias: Optional[bool] = True):
        if (size is None) == (scale_factor is None):
            raise ValueError("Only one of 'size' and 'scale_factor' parameters"
                             f" must be set")
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             f"Supported backends are 'cv2', 'pillow'")
        self.size = size
        self.scale_factor = scale_factor
        self.interpolation = interpolation
        self.backend = backend
        self.antialias = antialias

    def transform(self, results: dict) -> dict:
        img = results['img']
        h, w = dlicv.get_image_shape(img)
        img = dlicv.imresize(img, 
                             size=self.size,
                             scale_factor=self.scale_factor,
                             interpolation=self.interpolation,
                             backend=self.backend,
                             antialias=self.antialias)
        new_h, new_w = dlicv.get_image_shape(img)
        results['img'] = img
        results['img_shape'] = (new_h, new_w)
        results['scale_factor'] = (new_w / w, new_h / h)
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'interpolation={self.interpolation}'
        repr_str += f'backend={self.backend}, '
        repr_str += f'antialias={self.antialias})'
        return repr_str


class Pad(BaseTransform):
    """Pad the given image on all sides or to a certain shape with specified 
    padding mode and padding value.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Added Keys:

    - padding

    Args:
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
    """
    def __init__(self, 
                 padding: Optional[Union[int, Sequence[int]]] = None,
                 min_size: Optional[Union[int, Tuple[int, int]]] = None,
                 size_divisor: Optional[int] = None,
                 to_square: bool = False,
                 pad_position: Union[PositionType, str] = PositionType.AROUND,
                 pad_val: Union[Number, Sequence[Number]] = 0,
                 padding_mode: str = 'constant'):
        assert (min_size is None) or (size_divisor is None), \
            'only one of `size` and `size_divisor` can be set'
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if padding is not None:
            if (min_size is not None) or (size_divisor is not None
                                          ) or to_square:
                raise ValueError('`padding` can not be set together with '
                                 '`min_size`, `size_divisor` or `to_square`')
        pad_position = PositionType(pad_position)

        self.padding = padding
        self.min_size = min_size
        self.size_divisor = size_divisor
        self.to_square = to_square
        self.pad_position = pad_position
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    def transform(self, results: dict) -> dict:
        img = results['img']
        img, padding = dlicv.impad(img,
                                   padding=self.padding, 
                                   min_size=self.min_size,
                                   size_divisor=self.size_divisor,
                                   to_square=self.to_square,
                                   pad_position=self.pad_position,
                                   pad_val=self.pad_val,
                                   padding_mode=self.padding_mode,
                                   return_padding=True)
        h, w = dlicv.get_image_shape(img)
        results['img'] = img
        results['img_shape'] = (h, w)
        results['padding'] = padding
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(padding={self.padding}, '
        repr_str += f'min_size={self.min_size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'to_square={self.to_square}, '
        repr_str += f'pad_position={self.pad_position}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f"padding_mode='{self.padding_mode}')"
        return repr_str
        
         
class CenterCrop(BaseTransform):
    pass


class Flip(BaseTransform):
    """Flip the. Added or Updated
    keys: flip, flip_direction, img. 

    Required Keys:

    - img

    Modified Keys:

    - img

    Added Keys:

    - flip
    - flip_direction

    Args:
        direction(str | list[str]): The flipping direction. Options
            If input is a list, the length must equal ``prob``. Each
            element in ``prob`` indicates the flip probability of
            corresponding direction. Defaults to 'horizontal'.
    """

    def __init__(self, direction: str = 'horizontal') -> None:

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        assert isinstance(direction, str) and direction in valid_directions
        self.direction = direction

    def transform(self, results: dict) -> dict:
        """Transform function to flip images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'flip', and 'flip_direction' keys are
            updated in result dict.
        """
        # flip image
        results['img'] = dlicv.imflip(
            results['img'], direction=self.direction)

        results['flip'] = True
        results['flip_direction'] = self.direction

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'direction={self.direction})'

        return repr_str
    