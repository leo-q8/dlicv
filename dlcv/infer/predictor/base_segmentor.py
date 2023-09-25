from typing import List, Callable, Sequence, Optional

from torch import Tensor
import torch.nn.functional as F

from dlcv.structures import SegDataSample, PixelData
from .base import BasePredictor, ModelType

SampleList = List[SegDataSample]


class BaseSegmentor(BasePredictor):
    def __init__(self, 
                 backend_model: ModelType, 
                 pipeline: Sequence[Callable],
                 binary_thres: float = 0.3,
                 align_corners: bool = False):
        super().__init__(backend_model, pipeline)
        assert 0. < binary_thres < 1.
        self.binary_thres = binary_thres
        self.align_corners = align_corners
    
    def mask_postprocess(self,
                         seg_logits: Tensor,
                         img_meta: dict) -> Sequence:
        C, H, W = seg_logits.shape

        # remove padding area.
        padding = img_meta.get('padding')
        if padding is not None:
            left_padding, top_padding, right_padding, bottom_padding = padding
            seg_logits = seg_logits[..., top_padding: H - bottom_padding,
                                    left_padding: W - right_padding]
        
        # resize as original shape
        scale_factor = img_meta.get('scale_factor')
        if scale_factor is not None:
            ori_shape = img_meta.get('ori_shape')
            seg_logits = F.interpolate(
                seg_logits[None], ori_shape, mode='bilinear', 
                align_corners=self.align_corners)[0]
        
        if C > 1: 
            seg_pred = seg_logits.argmax(dim=0, keepdim=True)
        else:
            seg_pred = (seg_logits > self.binary_thres).to(seg_logits)

        return seg_logits, seg_pred

    def postprocess(self,
                    seg_logits: Tensor, 
                    batch_datasamples: SampleList, 
                    **kwargs) -> SampleList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_datasamples
        ]
        for data_sample, seg_logit, img_meta in \
                zip(batch_datasamples, seg_logits, batch_img_metas):
            seg_logit, seg_pred = self.mask_postprocess(
                seg_logit, img_meta)
            data_sample.seg_logits = PixelData(**{'data': seg_logit})
            data_sample.pred_sem_seg = PixelData(**{'data': seg_pred})
        return batch_datasamples