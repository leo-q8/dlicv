from typing import Optional, Sequence, Tuple
from numbers import Number

import numpy as np


def shift_scale_boxes(
    boxes: np.ndarray, 
    scale_factor: Optional[Tuple[float, float]],
    shift: Optional[Tuple[Number, Number]]
) -> np.ndarray:
    if shift is not None:
        boxes += shift * 2
    if scale_factor is not None:
        boxes = boxes * (scale_factor * 2)
    return boxes
    

def clip_boxes(boxes: np.ndarray, shape: Tuple[Number, Number]):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def get_box_wh(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w = boxes[:, 2] - boxes[: 0]
    h = boxes[:, 3] - boxes[: 1]
    return w, h

    
def batched_nms(boxes: np.ndarray,
                scores: np.ndarray,
                idxs: np.ndarray,
                score_thres: float = 0.25,
                iou_thres: float = 0.45,
                class_agnostic: bool = False,
                nms_pre: int = 30000) -> Tuple[np.ndarray, np.ndarray, 
                                               np.ndarray]:
    pass