from dlicv.structures import BaseDataElement
from .base import BaseTransform


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