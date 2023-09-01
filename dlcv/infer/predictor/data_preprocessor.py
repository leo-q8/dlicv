from numbers import Number
from typing import Sequence, Union, Optional


class ImgDataPreprocessor:
    def __init__(self,
                 mean: Optional[Union[Number, Sequence[Number]]] = None,
                 std: Optional[Union[Number, Sequence[Number]]] = None,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False):
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        assert (mean is None) == (std is None), (
            'mean and std should be both None or tuple')
        if mean is not None:
            if isinstance(mean, Number):
                mean = [mean]
            assert len(mean) == 3 or len(mean) == 1, (
                '`mean` should have 1 or 3 values, to be compatible with '
                f'RGB or gray image, but got {len(mean)} values')
            if isinstance(std, Number):
                mean = [std]
            assert len(std) == 3 or len(std) == 1, (  # type: ignore
                '`std` should have 1 or 3 values, to be compatible with RGB '  # type: ignore # noqa: E501
                f'or gray image, but got {len(std)} values')  # type: ignore
            self._enable_normalize = True
        self.mean = mean
        self.std = std
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
    
    def __call__(self, data_batch: Sequence[dict]) -> Union[dict, list]:
        _batch_inputs = []
        for data in data_batch:
            _data_input = data['inputs']
             # channel transform
            if self._channel_conversion:
                _data_input = _data_input[[2, 1, 0], ...]
                 # Convert to float after channel conversion to ensure
                # efficiency
                _data_input = _data_input.astype(float)
                # Normalization.
                if self._enable_normalize:
                    if len(self.mean) == 3:
                        assert len(_data_input.shape
                        ) == 3 and _data_input.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_data_input.shape}')
                    _data_input = (_data_input - self.mean) / self.std
                _batch_inputs.append(_data_input)


            
        