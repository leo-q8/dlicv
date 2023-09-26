from numbers import Number
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

from .base import BaseTransform


class ImgToTensor(BaseTransform):
    def __init__(self,
                 device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None) -> None:
        self.device = device 
        self.dtype = dtype

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
        results['img'] = img
        return results
    

class Normalize(BaseTransform):
    def __init__(self,
                 mean: Sequence[Number],
                 std: Sequence[Number],
                 to_rgb: bool = False) -> None:
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb
    
    def _normalize_tensor_img(self, img: torch.Tensor) -> torch.Tensor:
        if self.to_rgb:
            img = img.flip(-3)
        img = img.to(torch.float32)
        return TF.normalize(img, self.mean, self.std, inplace=False)
    
    def _normalize_array_img(self, img: np.ndarray) -> np.ndarray:
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        img = img.copy().astype(np.float32)
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if self.to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img

    def transform(self, results: dict) -> dict:
        img = results['img']
        if isinstance(img, np.ndarray):
            img = self._normalize_array_img(img)
        else:
            img = self._normalize_tensor_img(img)
        results['img'] = img
        return results



    