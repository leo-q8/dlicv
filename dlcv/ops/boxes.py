from typing import Optional, Tuple, Union
from numbers import Number

import numpy as np
import torch
from torch import Tensor
import torchvision

BoxType = Union[np.ndarray, Tensor]


def shift_scale_boxes(
    boxes: Tensor, 
    shift: Optional[Tuple[Number, Number]] = None,
    scale_factor: Optional[Tuple[float, float]] = None,
) -> Tensor:
    if shift is not None:
        boxes += boxes.new_tensor(shift * 2)
    if scale_factor is not None:
        boxes *= boxes.new_tensor(scale_factor * 2)
    return boxes
    

def clip_boxes_(boxes: BoxType, shape: Tuple[int, int]) -> None:
    if isinstance(Tensor):
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def clip_boxes(boxes: BoxType,
               shape: Tuple[int, int]) -> BoxType:
    if isinstance(boxes, Tensor):
        cmin = boxes.new_empty(boxes.shape[-1])
        cmin[0::2] = shape[1]
        cmin[1::2] = shape[0]
        clipped_boxes = torch.minimum(boxes, cmin)
        clipped_boxes.clamp_(0)
    else:
        cmin = np.empty(boxes.shape[-1], dtype=boxes.dtype)
        cmin[0::2] = shape[1]
        cmin[1::2] = shape[0]
        clipped_boxes = np.maximum(np.minimum(boxes, cmin), 0)
    return clipped_boxes


def resize_boxes(boxes: BoxType,
                 scale_factor: Tuple[int, int]) -> BoxType:
    assert len(scale_factor) == 2
    ctrs = (boxes[..., 2:] + boxes[..., :2]) / 2
    wh = boxes[..., 2:] - boxes[..., :2] + 1
    if isinstance(boxes, Tensor):
        scale_factor = boxes.new_tensor(scale_factor)
        wh = wh * scale_factor
        xy1 = ctrs - 0.5 * wh
        xy2 = ctrs + 0.5 * wh
        return torch.cat([xy1, xy2], dim=-1)
    else:
        wh = wh * scale_factor
        xy1 = ctrs - 0.5 * wh
        xy2 = ctrs + 0.5 * wh
        return np.concatenate([xy1, xy2], axis=-1)


def box_wh(boxes: BoxType) -> tuple:
    w = boxes[:, 2] - boxes[:, 0] 
    h = boxes[:, 3] - boxes[:, 1]
    return w, h


def box_area(boxes: BoxType) -> BoxType:
    w, h = box_wh(boxes)
    return w * h


def box_iou(boxes1: BoxType, boxes2: BoxType, eps=1e-7):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    if isinstance(boxes1, Tensor):
        tl1, rb1  = boxes1.unsqueeze(1).chunk(2, 2)
        tl2, rb2  = boxes2.unsqueeze(0).chunk(2, 2) 
        inter = (torch.min(rb1, rb2) - torch.max(tl1, tl2)).clamp_(0).prod(2)
    else:
        tl1, rb1  = boxes1[:, None, :2], boxes1[:, None, 2:]
        tl2, rb2  = boxes2[None, :, :2], boxes2[None, :, 2:]
        inter = (np.minimum(rb1, rb2) - np.maximum(tl1, tl2)).clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter + eps)


def box_ioa(boxes1: BoxType, boxes2: BoxType, eps=1e-7):
    area = box_area(boxes2)
    if isinstance(boxes1, Tensor):
        tl1, rb1  = boxes1.unsqueeze(1).chunk(2, 2)
        tl2, rb2  = boxes2.unsqueeze(0).chunk(2, 2) 
        inter = (torch.min(rb1, rb2) - torch.max(tl1, tl2)).clamp_(0).prod(2)
    else:
        tl1, rb1  = boxes1[:, None, :2], boxes1[:, None, 2:]
        tl2, rb2  = boxes2[None, :, :2], boxes2[None, :, 2:]
        inter = (np.minimum(rb1, rb2) - np.maximum(tl1, tl2)).clip(0).prod(2)
    return inter / (area + eps)
        
    
def batched_nms(boxes: Tensor,
                scores: Tensor,
                idxs: Tensor,
                iou_thres: float = 0.45,
                class_agnostic: bool = False,
                nms_pre: int = -1) -> Tuple[Tensor, Tensor]:

    if nms_pre > -1:
        scores, inds = scores.sort(descending=True)
        inds = inds[:nms_pre]
        scores, boxes, idxs = scores[inds], boxes[inds], idxs[inds]
    else:
        inds = torch.arange(len(scores)).to(idxs)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (
            max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
    keep_idxs = torchvision.ops.nms(boxes_for_nms, scores, iou_thres)

    boxes = torch.cat(boxes[keep_idxs], scores[keep_idxs, None], -1)
    inds = inds[keep_idxs]
    return boxes, inds