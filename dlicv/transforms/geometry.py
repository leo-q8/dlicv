from numbers import Number
from typing import Optional, Union, Tuple

import numpy as np

from dlicv.ops.image import imresize, impad, get_image_shape
from .base import BaseTransform


class Resize(BaseTransform):
    def __init__(self, 
                 size: Union[int, Tuple[int, int]],
                 keep_ratio: bool = False,
                 interpolation: str = 'bilinear'):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        
    def transform(self, results: dict) -> dict:
        img = results['img']
        h, w = get_image_shape(img)
        img = imresize(img, 
                       self.size,
                       self.keep_ratio,
                       self.interpolation)
        new_h, new_w = get_image_shape(img)
        results['img'] = img
        results['img_shape'] = (new_h, new_w)
        results['scale_factor'] = (new_w / w, new_h / h)
        return results


class Pad(BaseTransform):
    def __init__(self, 
                 size: Optional[Tuple[int, int]] = None,
                 size_divisor: Optional[int] = None,
                 pad_val: Number = 0,
                 mode: str = 'around',
                 to_square: bool = False,
                 padding_mode: str = 'constant') -> None:

        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.to_square = to_square
        if to_square:
            assert size is None, \
                'The size must be None when pad2square is True'
        else:
            assert (size is None) ^ (size_divisor is None), \
                'only one of size and size_divisor should be valid'
        assert mode in ['around', 'rb']
        self.mode = mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding_mode = padding_mode

    def transform(self, results: dict) -> dict:
        img = results['img']
        img, padding = impad(img,
                             self.size, 
                             self.size_divisor,
                             self.pad_val,
                             self.mode,
                             self.to_square,
                             self.padding_mode,
                             return_padding=True)
        h, w = get_image_shape(img)
        results['img'] = img
        results['img_shape'] = (h, w)
        results['padding'] = padding

        return results
        
         
    