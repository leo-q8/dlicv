import copy
from numbers import Number
from typing import Sequence, Union, List, Optional

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

from .base import BaseTransform
from .wrappers import Compose


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
                 mean: Union[Number, Sequence[Number]],
                 std: Union[Number, Sequence[Number]]) -> None:
        if isinstance(mean, (int, float)):
            mean = [mean]
        if isinstance(std, (int, float)):
            std = [std]
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


class TestTimeAug(BaseTransform):
    """Test-time augmentation transform.

    An example use is as followed:

    .. code-block::

        TestTimeAug(
             transforms=[
                [Resize(scale=1024, keep_ratio=True),
                 Flip(direction='vertical'),
                 Pad(to_square=True),
                 PackDetInputs(
                      meta_keys=('img_id', 'img_path', 'ori_shape', 
                                 'img_shape', 'scale_factor', 
                                 'flip_direction'))
                ],
                [Resize(scale=768, keep_ratio=True),
                 Flip(),
                 Pad(min_size=1024, to_square=True),
                 PackDetInputs(
                      meta_keys=('img_id', 'img_path', 'ori_shape', 
                                 'img_shape', 'scale_factor', 
                                 'flip_direction'))
                ],
                [Resize(scale=512, keep_ratio=True),
                 Pad(min_size=1024, to_square=True),
                 PackDetInputs(
                      meta_keys=('img_id', 'img_path', 'ori_shape', 
                                 'img_shape', 'scale_factor', 
                                 'flip_direction'))
                ]])

    ``results`` will be transformed using all transforms defined in
    ``transforms`` arguments.

    For the above configuration, there are four combinations of resize
    and flip:

    - Resize to (1333, 400) + no flip
    - Resize to (1333, 400) + flip
    - Resize to (1333, 800) + no flip
    - resize to (1333, 800) + flip

    After that, results are wrapped into lists of the same length as below:

    .. code-block::

        dict(
            inputs=[...],
            data_samples=[...]
        )

    The length of ``inputs`` and ``data_samples`` are both 4.

    Required Keys:

    - Depending on the requirements of the ``transforms`` parameter.

    Modified Keys:

    - All output keys of each transform.

    Args:
        transforms (list[list[dict]]): Transforms to be applied to data sampled
            from dataset. ``transforms`` is a list of list, and each list
            element usually represents a series of transforms with the same
            type and different arguments. Data will be processed by each list
            elements sequentially. See more information in :meth:`transform`.
    """

    def __init__(self, transforms: list, 
                 last_transforms: Optional[Union[callable, List[callable
                                                                ]]] = None):
        assert type(transforms) == list, '`transforms` must be a list'
        for transform_list in transforms:
            assert type(transform_list) == list, '`subroutine` must be a list'
            for transform in transform_list:
                if not callable(transform):
                    raise TypeError('transform must be callable, but '
                                    f'got {type(transform)}')
        if last_transforms is not None and type(last_transforms) == list:
            last_transforms = Compose(last_transforms)
        self.last_transforms = last_transforms
        self.subroutines = [
            Compose(subroutine) for subroutine in transforms
        ]

    def transform(self, results: dict) -> List[dict]:
        """Apply all transforms defined in :attr:`transforms` to the results.

        As the example given in :obj:`TestTimeAug`, ``transforms`` consists of
        2 ``Resize``, 2 ``RandomFlip`` and 1 ``PackDetInputs``.
        The data sampled from dataset will be processed as follows:

        1. Data will be processed by 2 ``Resize`` and return a list
           of 2 results.
        2. Each result in list will be further passed to 2
           ``RandomFlip``, and aggregates into a list of 4 results.
        3. Each result will be processed by ``PackDetInputs``, and
           return a list of dict.
        4. Aggregates the same fields of results, and finally returns
           a dict. Each value of the dict represents 4 transformed
           results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: The augmented data, where each value is wrapped
            into a list.
        """
        results_list = []  # type: ignore
        ori_img = results.pop('ori_img') if 'ori_img' in results else None
        for subroutine in self.subroutines:
            copy_result = copy.deepcopy(results)
            if ori_img is not None:
                copy_result['ori_img'] = ori_img
            result = subroutine(copy_result)
            assert isinstance(result, dict), (
                f'Data processed by {subroutine} must return a dict, but got '
                f'{result}')
            assert result is not None, (
                f'Data processed by {subroutine} in `TestTimeAug` should not '
                f'be None! Please check the transforms in {subroutine}')
            results_list.append(result)
        
        if self.last_transforms is not None:
            results_list = [self.last_transforms(result) for result in 
                            results_list]

        return results_list

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += 'transforms=\n'
        for subroutine in self.subroutines:
            repr_str += f'{repr(subroutine)}\n'
        return repr_str