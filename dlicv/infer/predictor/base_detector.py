from abc import ABCMeta, abstractmethod
from typing import Any, List, Callable, Optional, Sequence, Tuple

from torch import Tensor

from dlicv.structures import DetDataSample, InstanceData
from dlicv.ops.boxes import batched_nms, box_wh
from .base import BasePredictor, ModelType

SampleList = List[DetDataSample]
InstanceList = List[InstanceData]


class BaseDetector(BasePredictor, metaclass=ABCMeta):
    def __init__(self, 
                 backend_model: ModelType, 
                 pipeline: Sequence[Callable],
                 score_thres: Optional[float] = None,
                 nms_cfg: Optional[dict] = None,
                 min_box_size: int = -1,
                 max_det: int = -1):
        valid_nms_params = {'iou_thres', 'class_agnostic', 'nms_pre'} 
        if nms_cfg is not None:
            assert set(nms_cfg) <= valid_nms_params , \
                f'{set(nms_cfg) - valid_nms_params} is invalid nms params'
        super().__init__(backend_model, pipeline)
        if score_thres is not None:
            assert 0. < score_thres < 1.
        self.score_thres = score_thres
        self.min_box_size = min_box_size
        self.nms_cfg = nms_cfg
        self.max_det = max_det
    
    @abstractmethod
    def _parse_preds(self, preds: Any, 
                     batch_img_metas: Optional[list] = None) -> Sequence:
        pass
    
    def bbox_postprocess(self,
                         bboxes: Tensor,
                         scores: Tensor,
                         labels: Tensor,
                         img_meta: dict) -> Tuple[Tensor, Tensor, Tensor]:
        # filter low confidence boxes 
        if self.score_thres is not None:
            idxs = scores > self.score_thres
            bboxes, labels, scores = bboxes[idxs], labels[idxs], scores[idxs]

        # shift and rescale boxes to original image space.
        padding = img_meta.get('padding')
        if padding is not None:
            left_padding, top_padding = padding[:2]
            bboxes -= bboxes.new_tensor([left_padding, top_padding] * 2)
        scale_factor = img_meta.get('scale_factor')
        if scale_factor is not None:
            scale_factor = [1 / s for s in scale_factor]
            bboxes *= bboxes.new_tensor(scale_factor * 2)
        
        # filter samll size bboxes
        if self.min_box_size >= 0:
            w, h = box_wh(bboxes)
            valid_mask = (w > self.min_box_size) & (h > self.min_box_size)
            if not valid_mask.all():
                bboxes = bboxes[valid_mask]
                scores = scores[valid_mask]
                labels = labels[valid_mask]

        # do nms
        if self.nms_cfg is not None:
            keep_idxs = batched_nms(bboxes, scores, labels, **self.nms_cfg)
            if self.max_det > -1:
                keep_idxs = keep_idxs[:self.max_det]
            bboxes = bboxes[keep_idxs]
            scores = scores[keep_idxs]
            labels = labels[keep_idxs]

        # select topk
        elif self.max_det > -1:
            scores, idxs = scores.sort(descending=True)
            scores = scores[:self.max_det]
            bboxes = bboxes[idxs[:self.max_det]]
            labels = scores[idxs[:self.max_det]]
    
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

            bboxes, scores, labels = self.bbox_postprocess(
                bbox_preds, cls_socres, labels, img_meta)
            results = InstanceData()
            results.bboxes = bboxes
            results.scores = scores
            results.labels = labels
            data_sample.pred_instances = results
        return batch_datasamples
        
