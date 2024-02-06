from dlicv.structures import DetDataSample, SegDataSample
from .base import BaseTransform


class PackDetInputs(BaseTransform):
    def __init__(self,
                 meta_keys=('ori_shape', 'img_shape', 
                            'scale_factor', 'padding')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            packed_results['inputs'] = img

        data_sample = DetDataSample()
        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`. ' \
                f'The valid keys are {self.meta_keys}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


class PackSegInputs(BaseTransform):
    def __init__(self,
                 meta_keys=('ori_shape', 'img_shape', 
                            'scale_factor', 'padding')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            packed_results['inputs'] = img
        if 'ori_img' in results:
            packed_results['ori_img'] = results['ori_img']

        data_sample = SegDataSample()
        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`. ' \
                f'The valid keys are {self.meta_keys}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str