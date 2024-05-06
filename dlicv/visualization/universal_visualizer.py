from typing import List, Optional, Tuple, Union, Sequence

import cv2
import numpy as np
from numpy import ndarray
import torch

from dlicv.ops import bitmap_to_polygon, imresize
from dlicv.structures import InstanceData, PixelData, ClsDataSample
from .color import color_val
from .palette import _get_adaptive_scales, get_palette, jitter_color
from .utils import tensor2ndarray
from . import Visualizer


class UniversalVisualizer(Visualizer):
    """Universal Visualizer for multiple Computer Vision tasks.

    Args:
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        mask_color (str, tuple(int), optional): Color of masks.
            The tuple of color should be in BGR order.  Defaults to None.
        text_color (str, tuple(int), optional): Color of texts.
            The tuple of color should be in BGR order.
            Defaults to (200, 200, 200).
        kpt_color (str, tuple(tuple(int)), optional): Color of keypoints.
            The tuple of color should be in BGR order. Defaults to ``'red'``
        link_color (str, tuple(tuple(int)), optional): Color of skeleton.
            The tuple of color should be in BGR order. Defaults to ``None``
        skeleton (list, tuple): A list or tuple of key points index pair 
            which defined how key points are connected.
            Defaults to None.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        radius (int, float): The radius of keypoints. Defaults to 4.
        show_keypoint_weight (bool): Whether to adjust the transparency
            of keypoints according to their score. Defaults to ``False``.
        alpha (int, float): The transparency of bboxes or mask.
            Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from dlicv.structures import InstanceData
        >>> from dlicv.structures import DetDataSample
        >>> from dlicv.visualization import UniversalVisualizer

        >>> det_visualizer = UniversalVisualizer()
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> pred_instances = InstanceData()
        >>> pred_instances.bboxes = torch.Tensor([[1, 2, 2, 5]])
        >>> pred_instances.labels = torch.randint(0, 2, (1,))
        >>> pred_datasample = DetDataSample()
        >>> pred_datasample.pred_instances = pred_instances
        >>> det_visualizer.draw_instances(image, pred_datasample)
        >>> pred_instances = InstanceData()
    """
    DEFAULT_CLS_TEXT_CFG = {
        'family': 'monospace',
        'color': 'white',
        'bbox': dict(facecolor='black', alpha=0.5, boxstyle='Round'),
        'verticalalignment': 'top',
        'horizontalalignment': 'left',
    }

    def __init__(self,
                 image: Optional[np.ndarray] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (200, 200, 200),
                 kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = 'red',
                 link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
                 skeleton: Optional[Union[List, Tuple]] = None,
                 line_width: Union[int, float] = 3,
                 radius: Union[int, float] = 3,
                 show_keypoint_weight: bool = False,
                 alpha: float = 0.8) -> None:
        super().__init__(image=image)
        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.text_color = text_color
        self.mask_color = mask_color
        self.skeleton = skeleton
        self.radius = radius
        self.line_width = line_width
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight
    
    def draw_instances_bboxes(self, 
                              image: np.ndarray,
                              instances: InstanceData,
                              classes: Optional[Sequence[str]] = None,
                              palette: Optional[Union[Sequence[tuple
                                                               ], str]] = None,
                              boxes_line_width: Optional[Union[int, float
                                                               ]] = None,
                              show_label: bool = True,
                              show_score: bool = True) -> np.ndarray:
        self.set_image(image)
        boxes_line_width = self.line_width if boxes_line_width is None else \
            boxes_line_width

        if 'bboxes' in instances and instances.bboxes.sum() > 0:
            bboxes = instances.bboxes
            labels = instances.labels if 'labels' in instances else []

            max_label = int(max(labels) if len(labels) > 0 else 0)
            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]
            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=boxes_line_width)
            
            if (show_label and 'labels' in instances) or (show_score and 
                'scores' in instances):

                text_palette = get_palette(self.text_color, max_label + 1)
                if len(labels) == 0:
                    labels = [0] * len(bboxes)
                text_colors = [text_palette[label] for label in labels]

                positions = bboxes[:, :2] + boxes_line_width
                areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                    bboxes[:, 2] - bboxes[:, 0])
                areas = tensor2ndarray(areas) 
                scales = _get_adaptive_scales(areas)

                for i, pos in enumerate(positions):
                    label_text, score_text = '', ''
                    if show_label and 'labels' in instances:
                        if 'label_names' in instances:
                            label_text = instances.label_names[i]
                        elif classes is not None:
                            label_text = classes[labels[i]]
                        else:
                            label_text = f'class {labels[i]}'

                    if show_score and 'scores' in instances:
                        score = round(float(instances.scores[i]) * 100, 1)
                        score_text = f'{score}'

                    label_text = label_text + (': ' + score_text if label_text 
                        and score_text else score_text)
                    
                    self.draw_texts(
                        label_text,
                        pos,
                        colors=text_colors[i],
                        font_sizes=int(13 * scales[i]),
                        bboxes=[{
                            'facecolor': 'black',
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])

        return self.get_image()
    
    def draw_instances_kpts(self,
                            image: np.ndarray,
                            instances: InstanceData,
                            kpt_thr: float = 0.3,
                            skeleton_line_width: Optional[Union[int, float
                                                                ]] = None,
                            show_skeleton: bool = True,
                            show_kpt_idx: bool = False):
        self.set_image(image)
        skeleton_line_width = self.line_width if skeleton_line_width is None \
            else skeleton_line_width
        img_h, img_w, _ = image.shape
        
        if 'keypoints' in instances:
            keypoints = instances.get('transformed_keypoints',
                                      instances.keypoints)
            if isinstance(keypoints, torch.Tensor):
                keypoints = keypoints.cpu().numpy()

            if 'keypoints_visible' in instances:
                keypoints_visible = instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            for kpts, visible in zip(keypoints, keypoints_visible):
                kpts = np.array(kpts, copy=False)

                if self.kpt_color is None or isinstance(self.kpt_color, str):
                    kpt_color = [self.kpt_color] * len(kpts)
                elif len(self.kpt_color) == len(kpts):
                    kpt_color = self.kpt_color
                else:
                    raise ValueError(
                        f'the length of kpt_color '
                        f'({len(self.kpt_color)}) does not matches '
                        f'that of keypoints ({len(kpts)})')

                # draw links
                if show_skeleton and self.skeleton is not None and \
                    self.link_color is not None:
                    if self.link_color is None or isinstance(
                            self.link_color, str):
                        link_color = [self.link_color] * len(self.skeleton)
                    elif len(self.link_color) == len(self.skeleton):
                        link_color = self.link_color
                    else:
                        raise ValueError(
                            f'the length of link_color '
                            f'({len(self.link_color)}) does not matches '
                            f'that of skeleton ({len(self.skeleton)})')

                    for sk_id, sk in enumerate(self.skeleton):
                        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                        if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                                or pos1[1] >= img_h or pos2[0] <= 0
                                or pos2[0] >= img_w or pos2[1] <= 0
                                or pos2[1] >= img_h or visible[sk[0]] < kpt_thr
                                or visible[sk[1]] < kpt_thr
                                or link_color[sk_id] is None):
                            # skip the link that should not be drawn
                            continue

                        X = np.array((pos1[0], pos2[0]))
                        Y = np.array((pos1[1], pos2[1]))
                        color = link_color[sk_id]
                        if not isinstance(color, str):
                            color = tuple(int(c) for c in color)
                        transparency = self.alpha
                        if self.show_keypoint_weight:
                            transparency *= max(
                                0,
                                min(1,
                                    0.5 * (visible[sk[0]] + visible[sk[1]])))

                        self.draw_lines(
                            X, Y, color, line_widths=skeleton_line_width)

                # draw each point on image
                for kid, kpt in enumerate(kpts):
                    if visible[kid] < kpt_thr or kpt_color[kid] is None:
                        # skip the point that should not be drawn
                        continue

                    color = kpt_color[kid]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(0, min(1, visible[kid]))
                    self.draw_circles(
                        kpt,
                        radius=np.array([self.radius]),
                        face_colors=color,
                        edge_colors=color,
                        alpha=transparency,
                        line_widths=self.radius)
                    if show_kpt_idx:
                        kpt_idx_coords = kpt + [self.radius, -self.radius]
                        self.draw_texts(
                            str(kid),
                            kpt_idx_coords,
                            colors=color,
                            font_sizes=self.radius * 3,
                            vertical_alignments='bottom',
                            horizontal_alignments='center')

        return self.get_image()
           
    def draw_instances_masks(self, 
                            image: np.ndarray,
                            instances: InstanceData,
                            classes: Optional[Sequence[str]] = None,
                            palette: Optional[Union[Sequence[tuple
                                                             ], str]] = None,
                            mask_edge_width: Optional[Union[int, float
                                                            ]] = None,
                            show_mask: bool = True,
                            show_edge: bool = True,
                            show_label: bool = True,
                            show_score: bool = True) -> np.ndarray:
        self.set_image(image)
        mask_edge_width = self.line_width if mask_edge_width is None else \
            mask_edge_width

        if 'masks' in instances:
            assert show_mask or show_edge, ('One of `show_mask` and '
                                            '`show_edge` must be True!')
            masks = instances.masks
            labels = instances.labels if 'labels' in instances else []
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()

            masks = masks.astype(bool)

            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None \
                else self.mask_color
            mask_palette = get_palette(mask_color, max_label + 1)
            colors = [jitter_color(mask_palette[label]) for label in labels]
            if show_mask:
                self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)
            if show_edge:
                edge_colors = 'w' if show_mask else colors
                polygons = []
                for i, mask in enumerate(masks):
                    contours, _ = bitmap_to_polygon(mask)
                    polygons.extend(contours)
                self.draw_polygons(polygons, 
                                   edge_colors=edge_colors,
                                   alpha=self.alpha,
                                   line_widths=mask_edge_width)

            if (show_label and 'labels' in instances) or (show_score and 
                'scores' in instances):
                text_palette = get_palette(self.text_color, max_label + 1)
                if len(labels) == 0:
                    labels = [0] * len(masks)
                text_colors = [text_palette[label] for label in labels]

                areas = []
                positions = []
                for mask in masks:
                    _, _, stats, centroids = cv2.connectedComponentsWithStats(
                        mask.astype(np.uint8), connectivity=8)
                    if stats.shape[0] > 1:
                        largest_id = np.argmax(stats[1:, -1]) + 1
                        positions.append(centroids[largest_id])
                        areas.append(stats[largest_id, -1])
                areas = np.stack(areas, axis=0)
                scales = _get_adaptive_scales(areas)

                for i, pos in enumerate(positions):
                    label_text, score_text = '', ''
                    if show_label and 'labels' in instances:
                        if 'label_names' in instances:
                            label_text = instances.label_names[i]
                        elif classes is not None:
                            label_text = classes[labels[i]]
                        else:
                            label_text = f'class {labels[i]}'

                    if show_score and 'scores' in instances:
                        score = round(float(instances.scores[i]) * 100, 1)
                        score_text = f'{score}'

                    label_text = label_text + (': ' + score_text if label_text 
                        and score_text else score_text)
                    
                    self.draw_texts(
                        label_text,
                        pos,
                        colors=text_colors[i],
                        font_sizes=int(13 * scales[i]),
                        horizontal_alignments='center',
                        bboxes=[{
                            'facecolor': 'black',
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])

        return self.get_image()

    def draw_instances(self, image: np.ndarray, 
                       instances: InstanceData,
                       classes: Optional[Sequence[str]] = None,
                       palette: Optional[Union[Sequence[tuple], str]] = None,
                       boxes_line_width: Optional[Union[int, float]] = None,
                       skeleton_line_width: Optional[Union[int, float]] = None,
                       mask_edge_width: Optional[Union[int, float]] = None,
                       kpt_thr: float = 0.3,
                       show_bbox: bool = True,
                       show_label: bool = True,
                       show_score: bool = True,
                       show_mask: bool = True,
                       show_edge: bool = True,
                       show_kpt = False,
                       show_skeleton = False,
                       show_kpt_idx: bool = False) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        if show_bbox:
            image = self.draw_instances_bboxes(image, 
                                               instances, 
                                               classes,
                                               palette, 
                                               boxes_line_width,
                                               show_label, 
                                               show_score)
            #Prioritize drawing the labels and scores around the bboxes.
            show_score, show_label = False, False
        
        if show_mask or show_edge:
            image = self.draw_instances_masks(image, 
                                              instances,
                                              classes, 
                                              palette,
                                              mask_edge_width, 
                                              show_mask,
                                              show_edge,
                                              show_label, 
                                              show_score)

        if show_kpt:
            image = self.draw_instances_kpts(image, 
                                             instances,
                                             kpt_thr,
                                             skeleton_line_width,
                                             show_skeleton,
                                             show_kpt_idx)
        self.set_image(image) 
        return self.get_image()
    
    def draw_sem_seg(self, 
                     image: ndarray, 
                     sem_seg: PixelData,
                     classes: Optional[Sequence[str]] = None,
                     palette: Optional[Union[Sequence[tuple], str]] = None,
                     show_label: Optional[bool] = True) -> np.ndarray:
        sem_seg = sem_seg.cpu().data  
        ids = np.unique(sem_seg)[::-1]
        max_label = int(max(ids))
        mask_color = palette if self.mask_color is None \
                else self.mask_color
        mask_palette = get_palette(mask_color, max_label + 1)
        labels = np.array(ids, dtype=np.int64)

        colors = [mask_palette[label] for label in labels]
        mask = np.zeros_like(image, dtype=np.uint8)
        for label, color in zip(labels, colors):
            mask[sem_seg[0] == label, :] = color
        self.set_image(image)

        if show_label:
            def _get_center_loc(mask: np.ndarray) -> np.ndarray:
                """Get semantic seg center coordinate.

                Args:
                    mask: np.ndarray: get from sem_seg
                """
                loc = np.argwhere(mask == 1)

                loc_sort = np.array(
                    sorted(loc.tolist(), key=lambda row: (row[0], row[1])))
                y_list = loc_sort[:, 0]
                unique, indices, counts = np.unique(
                    y_list, return_index=True, return_counts=True)
                y_loc = unique[counts.argmax()]
                y_most_freq_loc = loc[loc_sort[:, 0] == y_loc]
                center_num = len(y_most_freq_loc) // 2
                x = y_most_freq_loc[center_num][1]
                y = y_most_freq_loc[center_num][0]
                return np.array([x, y])

            font = cv2.FONT_HERSHEY_SIMPLEX
            # (0,1] to change the size of the text relative to the image
            scale = 0.05
            fontScale = min(image.shape[0], image.shape[1]) / (25 / scale)
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]
            if image.shape[0] < 300 or image.shape[1] < 300:
                thickness = 1
                rectangleThickness = 1
            else:
                thickness = 2
                rectangleThickness = 2
            lineType = 2

            if isinstance(sem_seg[0], torch.Tensor):
                masks = sem_seg[0].cpu().numpy() == labels[:, None, None]
            else:
                masks = sem_seg[0] == labels[:, None, None]
            masks = masks.astype(np.uint8)
            for mask_num in range(len(labels)):
                classes_id = labels[mask_num]
                classes_color = colors[mask_num]
                loc = _get_center_loc(masks[mask_num])
                if classes is not None:
                    text = classes[classes_id]
                else:
                    text = f'class {classes_id}'
                (label_width, label_height), baseline = cv2.getTextSize(
                    text, font, fontScale, thickness)
                mask = cv2.rectangle(mask, loc, 
                                     (loc[0] + label_width + baseline, 
                                      loc[1] + label_height + baseline), 
                                     color_val(classes_color), -1)
                mask = cv2.rectangle(mask, loc, 
                                     (loc[0] + label_width + baseline, 
                                      loc[1] + label_height + baseline), 
                                     (0, 0, 0), rectangleThickness)
                mask = cv2.putText(mask, text, (loc[0], loc[1] + label_height),
                                   font, fontScale, text_colors[mask_num], 
                                   thickness, lineType)
        color_seg = (image * (1 - self.alpha) + mask * self.alpha).astype(
            np.uint8)
        self.set_image(color_seg)
        return color_seg

    def draw_cls(self, image: np.ndarray,
                 data_sample: ClsDataSample,
                 classes: Optional[Sequence[str]] = None,
                 show_score: bool = True,
                 resize: Optional[int] = None):

        if resize is not None:
            h, w = image.shape[:2]
            if w < h:
                image = imresize(image, (resize, resize * h // w))
            else:
                image = imresize(image, (resize * w // h, resize))

        texts = []
        self.set_image(image)

        if 'pred_label' in data_sample:
            idx = data_sample.pred_label.tolist()
            score_labels = [''] * len(idx)
            class_labels = [''] * len(idx)
            if show_score and 'pred_score' in data_sample:
                score_labels = [
                    f', {data_sample.pred_score[i].item():.2f}' for i in idx
                ]

            if classes is not None:
                class_labels = [f' ({classes[i]})' for i in idx]

            labels = [
                str(idx[i]) + score_labels[i] + class_labels[i]
                for i in range(len(idx))
            ]
            prefix = 'Prediction: '
            texts.append(prefix + ('\n' + ' ' * len(prefix)).join(labels))
        #Get adaptive scale according to image shape.
        #The target scale depends on the the short edge length of the image. 
        # If the short edge length equals 224, the output is 1.0.
        long_edge_length = max(image.shape[:2])
        img_scale = long_edge_length / 224.
        img_scale = min(max(img_scale, 0.3), 3.0)
        text_cfg = {
            'size': int(img_scale * 7),
            **self.DEFAULT_CLS_TEXT_CFG,
        }
        self.ax_save.text(
            img_scale * 5,
            img_scale * 5,
            '\n'.join(texts),
            **text_cfg,
        )
        drawn_img = self.get_image()
        return drawn_img