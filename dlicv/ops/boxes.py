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

def flip_boxes(boxes: BoxType,
               img_shape: Tuple[int],
               direction: str = 'horizontal') -> Tensor:
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 4*k)
        img_shape (Tuple[int]): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"

    Returns:
        Tensor: Flipped bboxes.
    """
    assert boxes.shape[-1] % 4 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = boxes.clone() if isinstance(boxes, Tensor) else np.copy(boxes)
    if direction == 'horizontal':
        flipped[..., 0::4] = img_shape[1] - boxes[..., 2::4]
        flipped[..., 2::4] = img_shape[1] - boxes[..., 0::4]
    elif direction == 'vertical':
        flipped[..., 1::4] = img_shape[0] - boxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - boxes[..., 1::4]
    else:
        flipped[..., 0::4] = img_shape[1] - boxes[..., 2::4]
        flipped[..., 1::4] = img_shape[0] - boxes[..., 3::4]
        flipped[..., 2::4] = img_shape[1] - boxes[..., 0::4]
        flipped[..., 3::4] = img_shape[0] - boxes[..., 1::4]
    return flipped


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
    if boxes.numel() == 0:
       return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    if nms_pre > -1:
        scores, inds = scores.sort(descending=True)
        inds = inds[:nms_pre]
        scores, boxes, idxs = scores[inds], boxes[inds], idxs[inds]
    else:
        inds = torch.arange(len(scores)).to(idxs)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # strategy: in order to perform NMS independently per class,
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (
            max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
    keep_idxs = torchvision.ops.nms(boxes_for_nms, scores, iou_thres)

    inds = inds[keep_idxs]
    return inds