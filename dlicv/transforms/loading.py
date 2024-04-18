from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import dlicv
from dlicv.utils import DeviceType
from .base import BaseTransform


class LoadImage(BaseTransform):
    """Load an image from file or ndarray.

    If the loaded image is a numpy array, then it will be saved in res dict 
    as is.

    Required Keys:

    - img_path
    or
    - ori_img

    Modified Keys:

    - img
    - ori_img
    - channel_order
    - img_shape
    - ori_shape

    Args:
        color_type (str): The flag argument for :func:`dlicv.imread`.
            Defaults to 'color'.
        channel_order:  Order of channel, candidates are `bgr` and `rgb`. The
            channel_order argument for :func:`dlicv.imread`
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`dlicv.imread`.
            Defaults to 'cv2'.
        to_float32 (bool): Whether to convert the loaded image to a float32.
            If set to False, the loaded image is uint8.
            Defaults to False.
        to_tensor (bool): Whether to convert the array image to 
            :obj:`torch.Tensor`
            Defaults to False.
        device (str | torch.device | None): The targer device to store tensor 
            image. Only valid when `to_tensor` is True.
    """
    def __init__(self,
                 imdecode_backend: str = 'cv2',
                 color_type: str = 'color',
                 channel_order: str = 'bgr',
                 to_float32: bool = False,
                 to_tensor: bool = False,
                 device: Optional[DeviceType] = None) -> None:
        self.imdecode_backend = imdecode_backend
        self.color_type = color_type
        self.channel_order = channel_order
        self.to_float32 = to_float32
        self.to_tensor = to_tensor
        assert to_tensor or device is None, \
            'device only valid when `to_tensor` is True'
        self.device = device
    
    def transform(self, results: dict) -> dict:
        res = dict()
        if 'img_path' in results:
            res['img_path'] = results['img_path']
            img = dlicv.imread(results['img_path'], 
                               flag=self.color_type,
                               channel_order=self.channel_order,
                               backend=self.imdecode_backend)
        elif 'ori_img' in results:
            img = results['ori_img']
        
        res['channel_order'] = self.channel_order
        res['ori_img'] = img
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
        res['img'] = img
        res['img_shape'] = shape
        res['ori_shape'] = shape
        return res

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'to_float32={self.to_float32}, '
                    f'to_tensor={self.to_tensor}, '
                    f'device={self.device})')
        return repr_str


class LoadImgFromNDArray(BaseTransform):
    """Load an image from ndarray.

    Required Keys:

    - ori_img

    Modified Keys:

    - img
    - ori_img
    - channel_order
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32.
            If set to False, the loaded image is uint8.
            Defaults to False.
        to_tensor (bool): Whether to convert the array image to 
            :obj:`torch.Tensor`
            Defaults to False.
        device (str | torch.device | None): The targer device to store tensor 
            image. Only valid when `to_tensor` is True.
        channel_order:  Order of original input array image's channel, 
            candidates are `bgr` and `rgb`. This parameter will be saved in 
            results dict as it.
    """
    def __init__(self, 
                 to_float32: bool = False,
                 to_tensor: bool = False,
                 device: Optional[str] = None,
                 channel_order: str = 'bgr'):
        self.to_float32 = to_float32
        self.to_tensor = to_tensor
        assert to_tensor or device is None, \
            'device only valid when `to_tesor` is True'
        self.device = device
        assert channel_order in ('bgr', 'rgb')
        self.channel_order = channel_order
    
    def transform(self, results: dict) -> dict:
        res = dict()
        img = results['ori_img']
        res['ori_img'] = img
        res['channel_order'] = self.channel_order
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
        res['img'] = img
        res['img_shape'] = shape
        res['ori_shape'] = shape
        return res
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, ',
        repr_str += f'to_tensor={self.to_tensor}, ',
        repr_str += f'device={self.device}, '
        repr_str += f"channel_order='{self.channel_order}')"
        return repr_str

            
class LoadImgFromTensor(BaseTransform):
    """Load an image from :obj:`torch.Tensor`.

    Required Keys:

    - ori_img

    Modified Keys:

    - img
    - ori_img
    - channel_order
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32.
            If set to False, the loaded image is uint8.
            Defaults to False.
        device (str | torch.device | None): The targer device to store tensor 
            image. Only valid when `to_tensor` is True.
    channel_order: Order of original input tensor image's channel, candidates 
        are `bgr` and `rgb`. This parameter will be saved in res dict as it.
    """ 
    def __init__(self, 
                 to_float32: bool = False,
                 device: Optional[str] = None,
                 channel_order = 'bgr'):
        self.to_float32 = to_float32
        self.device = device
        assert channel_order in ('bgr', 'rgb')
        self.channel_order = channel_order
    
    def transform(self, results: dict) -> dict:
        res = dict()
        img: torch.Tensor = results['ori_img']
        res['channel_order'] = self.channel_order
        if img.ndim < 2:
            raise ValueError
        elif img.ndim == 2:
            img = img.unsqueeze(0)
        res['ori_img'] = img
        if self.device is not None:
            img = img.to(self.device)
        if self.to_float32:
            img = img.to(torch.float32)
        size = tuple(img.shape[-2:])
        res['img'] = img
        res['img_shape'] = size
        res['ori_shape'] = size
        return res
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f'device={self.device}, '
        repr_str += f"channel_order='{self.channel_order}')"
        return repr_str