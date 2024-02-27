from typing import Union, Sequence

import numpy as np
import torch
from dlicv.structures import BaseDataElement

LABEL_TYPE = Union[torch.Tensor, np.ndarray, Sequence, int]
SCORE_TYPE = Union[torch.Tensor, np.ndarray, Sequence]


def format_label(value: LABEL_TYPE) -> torch.Tensor:
    """Convert various python types to label-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.

    Returns:
        :obj:`torch.Tensor`: The foramtted label tensor.
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


def format_score(value: SCORE_TYPE) -> torch.Tensor:
    """Convert various python types to score-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence): Score values.

    Returns:
        :obj:`torch.Tensor`: The foramtted score tensor.
    """

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not isinstance(value, str):
        value = torch.tensor(value).float()
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


class ClsDataSample(BaseDataElement):
    """A general data structure interface.

    It's used as the interface between different components.

    The following fields are convention names in MMPretrain, and we will set or
    get these fields in data transforms, models, and metrics if needed. You can
    also set any new fields for your need.

    Meta fields:
        img_shape (Tuple): The shape of the corresponding input image.
        ori_shape (Tuple): The original shape of the corresponding image.
        sample_idx (int): The index of the sample in the dataset.
        num_classes (int): The number of all categories.

    Data fields:
        gt_label (tensor): The ground truth label.
        gt_score (tensor): The ground truth score.
        pred_label (tensor): The predicted label.
        pred_score (tensor): The predicted score.

    Examples:
        >>> import torch
        >>> from mmpretrain.structures import DataSample
        >>>
        >>> img_meta = dict(img_shape=(960, 720), num_classes=5)
        >>> data_sample = DataSample(metainfo=img_meta)
        >>> data_sample.set_gt_label(3)
        >>> print(data_sample)
        <DataSample(
        META INFORMATION
            num_classes: 5
            img_shape: (960, 720)
        DATA FIELDS
            gt_label: tensor([3])
        ) at 0x7ff64c1c1d30>
        >>>
        >>> # For multi-label data
        >>> data_sample = DataSample().set_gt_label([0, 1, 4])
        >>> print(data_sample)
        <DataSample(
        DATA FIELDS
            gt_label: tensor([0, 1, 4])
        ) at 0x7ff5b490e100>
        >>>
        >>> # Set one-hot format score
        >>> data_sample = DataSample().set_pred_score([0.1, 0.1, 0.6, 0.1])
        >>> print(data_sample)
        <DataSample(
        META INFORMATION
            num_classes: 4
        DATA FIELDS
            pred_score: tensor([0.1000, 0.1000, 0.6000, 0.1000])
        ) at 0x7ff5b48ef6a0>
        >>>
        >>> # Set custom field
        >>> data_sample = DataSample()
        >>> data_sample.my_field = [1, 2, 3]
        >>> print(data_sample)
        <DataSample(
        DATA FIELDS
            my_field: [1, 2, 3]
        ) at 0x7f8e9603d3a0>
        >>> print(data_sample.my_field)
        [1, 2, 3]
    """

    @property
    def gt_label(self) -> LABEL_TYPE:
        return self._gt_label
    
    @gt_label.setter
    def gt_label(self, value: LABEL_TYPE):
        """Set ``gt_label``."""
        self.set_field(format_label(value), '_gt_label', dtype=torch.Tensor)
    
    @gt_label.deleter
    def gt_label(self):
        del self._gt_label

    @property
    def gt_class(self) -> str:
        return self._gt_class
    
    @gt_class.setter
    def gt_class(self, gt_class: str):
        self.set_field(gt_class, '_gt_class', dtype=str)

    @gt_class.deleter
    def gt_class(self):
        del self._gt_class

    @property
    def pred_label(self) -> LABEL_TYPE:
        return self._pred_label
    
    @pred_label.setter
    def pred_label(self, value: LABEL_TYPE):
        """Set ``gt_label``."""
        self.set_field(format_label(value), '_pred_label', dtype=torch.Tensor)
    
    @pred_label.deleter
    def pred_label(self):
        del self._pred_label

    @property
    def pred_score(self):
        return self._pred_score

    @pred_score.setter
    def pred_score(self, value: SCORE_TYPE) -> 'ClsDataSample':
        """Set ``pred_score``."""
        score = format_score(value)
        self.set_field(score, '_pred_score', dtype=torch.Tensor)
        if hasattr(self, 'num_classes'):
            assert len(score) == self.num_classes, \
                f'The length of score {len(score)} should be '\
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes', value=len(score), field_type='metainfo')
    
    @pred_score.deleter
    def pred_score(self):
        del self._pred_score
        del self.num_classes
    
    @property
    def pred_class(self) -> str:
        return self._pred_class
    
    @pred_class.setter
    def pred_class(self, pred_class: str):
        self.set_field(pred_class, '_pred_class', dtype=str)

    @pred_class.deleter
    def pred_class(self):
        del self._pred_class
    
    def __repr__(self) -> str:
        """Represent the object."""

        def dump_items(items, prefix=''):
            return '\n'.join(f'{prefix}{k}: {v}' for k, v in items)

        repr_ = ''
        if len(self._metainfo_fields) > 0:
            repr_ += '\n\nMETA INFORMATION\n'
            repr_ += dump_items(self.metainfo_items(), prefix=' ' * 4)
        if len(self._data_fields) > 0:
            repr_ += '\n\nDATA FIELDS\n'
            repr_ += dump_items(self.items(), prefix=' ' * 4)

        repr_ = f'<{self.__class__.__name__}({repr_}\n\n) at {hex(id(self))}>'
        return repr_
