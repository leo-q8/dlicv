from numbers import Number
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

from dlicv.utils.device import DeviceType
from .base import BaseTransform


class ImgToTensor(BaseTransform):
    """Convert image to :obj:`torch.Tensor`.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Required keys:

    - img

    Modified Keys:

    - img

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        device (str | None): 
    """
    def __init__(self,
                 to_float32: bool = False,
                 device: Optional[DeviceType] = None) -> None:
        self.device = device 
        self.to_float32 = to_float32

    def transform(self, results: dict) -> dict:
        img = results['img']
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        # To improve the computational speed by by 3-5 times, apply:
        # If image is not contiguous, use
        # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
        # If image is already contiguous, use
        # `torch.permute()` followed by `torch.contiguous()`
        # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
        # for more details
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            img = torch.from_numpy(img).to(self.device).to(self.dtype)
        else:
            img = torch.from_numpy(img).to(self.device).permute(
                2, 0, 1).contiguous().to(self.dtype)
        if self.to_float32:
            img = img.to(torch.float32)
        results['img'] = img
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, device={self.device})'
        return repr_str
    

class Normalize(BaseTransform):
    """Normalize the image.

    Required Keys:

    - img

    Modified Keys:

    - img

    Added Keys:

    - img_norm_cfg

      - mean
      - std

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
    """
    def __init__(self,
                 mean: Sequence[Number],
                 std: Sequence[Number]) -> None:
        self.mean = list(mean)
        self.std = list(std)
    
    def _normalize_tensor_img(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(torch.float32)
        return TF.normalize(img, self.mean, self.std, inplace=False)
    
    def _normalize_array_img(self, img: np.ndarray) -> np.ndarray:
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        img = img.astype(np.float32)
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1.0 / np.float64(std.reshape(1, -1))
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img

    def transform(self, results: dict) -> dict:
        img = results['img']
        if isinstance(img, np.ndarray):
            img = self._normalize_array_img(img)
        else:
            img = self._normalize_tensor_img(img)
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std)
        results['img'] = img
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std})'
        return repr_str