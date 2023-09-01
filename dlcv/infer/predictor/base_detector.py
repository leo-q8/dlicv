from abc import ABCMeta, abstractmethod
from typing import Any, List, Tuple, Callable, Union, Sequence, Optional

import numpy as np

from dlcv.structures import DetDataSample, InstanceData
from dlcv.ops.bbox import batched_nms, get_box_wh
from ..backend import BackendModel
from .base import BasePredictor

SampleList = List[DetDataSample]
InstanceList = List[InstanceData]


class BaseDetector(BasePredictor, metaclass=ABCMeta):
    def __init__(self, 
                 backend_model: Union[dict, BackendModel],
                 pipeline: Sequence[Callable],
                 nms_cfg: Optional[dict] = None,
                 min_box_size: int = -1,
                 max_det: int = -1):
        super().__init__(backend_model, pipeline)
        valid_nms_params = {'score_thres', 'iou_thres', 
                            'class_agnostic','nms_pre'} 
        assert set(nms_cfg) <= valid_nms_params , \
            f'{set(nms_cfg) - valid_nms_params} is invalid nms params'
        self.nms_cfg = nms_cfg
    
    @abstractmethod
    def _parse_preds(self, preds: Any, 
                     batch_img_metas: Optional[list]) -> Sequence:
        pass
    
    def _bbox_postprocess(self,
                          bboxes: np.ndarray,
                          scores: np.ndarray,
                          labels: np.ndarray,
                          img_meta: dict,
                          min_bbox_size: int = -1,
                          max_det: int = -1,
                          nms_cfg: Optional[dict] = None) -> Sequence:
        # shift and rescale boxes to original image space.
        padding = img_meta.get('padding')
        if padding is not None:
            left_padding, top_padding = padding[:2]
            bboxes -= [left_padding, top_padding] * 2
        scale_factor = img_meta.get('scale_factor')
        if scale_factor is not None:
            scale_factor = [1 / s for s in scale_factor]
            bboxes /= scale_factor * 2
        
        # filter samll size bboxes
        if min_bbox_size >= 0:
            w, h = get_box_wh(bboxes)
            valid_mask = (w > min_bbox_size) & (h > min_bbox_size)
            if not valid_mask.all():
                bboxes = bboxes[valid_mask]
                scores = scores[valid_mask]
                labels = labels[valid_mask]

        # do nms
        if nms_cfg is not None:
            bboxes, keep_idxs = batched_nms(bboxes, scores, 
                                            labels, **nms_cfg)
            if max_det > -1:
                bboxes = bboxes[:max_det]
                keep_idxs = keep_idxs[:max_det]
            scores = scores[keep_idxs]
            labels = labels[keep_idxs]
        # select topk
        elif max_det > -1:
            idxs = scores.argsort()[::-1][:max_det]
            bboxes = bboxes[idxs]
            scores = scores[idxs]
            labels = scores[idxs]
    
        return bboxes, scores, labels

    def postprocess(self,
                    preds: Any, 
                    batch_datasamples: SampleList, 
                    **kwargs) -> SampleList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_datasamples
        ]
        parsed_preds = self._parse_preds(preds, batch_img_metas)
        for data_sample, cls_socres, bbox_preds, labels, img_meta in \
                zip(batch_datasamples, *parsed_preds, batch_img_metas):
            results = InstanceData()
            results.bboxes = bbox_preds
            results.scores = cls_socres
            results.labels = labels
            results = self._bbox_postprocess(results, img_meta)
            data_sample.pred_instances = results
        return batch_datasamples
        
