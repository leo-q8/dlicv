from typing import Optional

import numpy as np
import torch

from dlicv.structures import BaseDataElement
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
            img = torch.from_numpy(img).to(self.device)
        else:
            img = torch.from_numpy(img).to(self.device).permute(
                2, 0, 1).contiguous()
        if self.to_float32 and img.dtype != torch.float32:
            img = img.to(torch.float32)
        results['img'] = img
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, device={self.device})'
        return repr_str


class PackImgInputs(BaseTransform):
    """Pack the inputs data for the classification / detection / 
    semantic segmentation.

    The ``img_meta`` item is always populated. The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``channel_order: Order of original input image's channel.

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``padding``: a tuple contain the padding size (pad_left, pad_top, 
            pad_right, pad_bottom)

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be collected in 
            ``data[img_metas]``.  Default: ``('img_path', 'ori_shape', 
            'img_shape', 'scale_factor', 'padding')``
    """
    def __init__(self,
                 datasample_type: type,
                 meta_keys=('img_path', 'ori_shape', 'channel_order', 
                            'img_shape', 'scale_factor', 'padding')):
        self.datasample_type = datasample_type
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'ori_imgs' (obj:`np.ndarray | obj:`torch.Tensor`): The original 
                image array or tensor. This is useful for visualization in 
                :class:`BasePredictor`.
            - 'data_sample' (obj:`BaseDataElement`): The meta info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            packed_results['inputs'] = results['img']
        if 'ori_img' in results:
            packed_results['ori_imgs'] = results['ori_img']

        data_sample: BaseDataElement = self.datasample_type()
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(datasample_type={self.datasample_type}, '
        repr_str += f'meta_keys={self.meta_keys})'
        return repr_str