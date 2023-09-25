from typing import Optional

import numpy as np
import torch

from .base import BaseTransform


class LoadImgFromNDArray(BaseTransform):
    def __init__(self, 
                 to_float32: bool = False,
                 to_tensor: bool = False,
                 device: Optional[str] = None):
        self.to_float32 = to_float32
        self.to_tensor = to_tensor
        assert to_tensor or device is None, \
            'device only valid when to_tesor is True'
        self.device = device
    
    def transform(self, results: dict) -> dict:
        img = results['inputs']
        shape = img.shape[:2]
        if self.to_tensor:
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = torch.from_numpy(img)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
                img = img.to(self.device)
            if self.to_float32:
                img = img.to(torch.float32)
        elif self.to_float32:
            img = img.astype(np.float32)
        results = dict()
        results['img'] = img
        results['img_shape'] = shape
        results['ori_shape'] = shape
        return results

            
class LoadImgFromTensor(BaseTransform):
    def __init__(self, 
                 to_float32: bool = False,
                 device: Optional[str] = None):
        self.to_float32 = to_float32
        self.device = device
    
    def transform(self, results: dict) -> dict:
        img = results['inputs']
        if self.device is not None:
            img = img.to(self.device)
        if self.to_float32:
            img = img.to(torch.float32)
        size = tuple(img.shape[:2])
        results['img'] = img
        results['img_shape'] = size
        results['ori_shape'] = size
        return results