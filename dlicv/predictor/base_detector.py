from abc import abstractmethod
import os.path as osp
from typing import Any, List, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from pathlib import Path
from torch import Tensor

from dlicv.structures import DetDataSample, InstanceData
from dlicv.ops import batched_nms, box_wh, flip_boxes, imwrite
from dlicv.visualization import UniversalVisualizer
from dlicv.utils import Classes
from dlicv.transforms import Compose, PackImgInputs, TestTimeAug
from .base import BasePredictor, ModelType

SampleList = List[DetDataSample]
InstanceList = List[InstanceData]
ImgsType = List[Union[np.ndarray, Tensor]]


class BaseDetector(BasePredictor):
    """Base Object Detection Predictor.

    Args:
        backend_model (dict | torch.nn.Module | BackendModel): A `BackendModel` 
            or `torch.nn.Module` to execute the inference. The `dict` param is 
            for initialing the `BackendModel`. 
        pipeline (Callable | Sequence[Callable]): Data preprocess pipeline. A 
            compose of series of data transformation defined in 
            `dlicv.trasnforms`.
        conf: (float | None): Object confidence threshold for detection. 
            Objects with a score lower than `conf` will be filtered out in 
            :meth:`bbox_postprocess`. Defaults to None.
        nms_cfg: (dict | None): A dict that contains the arguments of 
            non-maximum suppression (NMS) operations. If specified, a NMS 
            operation will be executed in :meth:`bbox_postprocess` before 
            return boxes. Defaults to None.
            Valid keys are:

                - iou_thres (float): IoU threshold to be considered as 
                    conflicted.

                - class_agnostic (bool): Whether generate class agnostic 
                    prediction. If True, the model is agnostic to the number of 
                    classes, and all classes will be considered as one.

                - nms_pre (int): The maximum number of boxes into NMS 
                    opreation.

        min_box_size (int): The minimum box width and height in pixels. If > 0,
            the bboxes small than it will be filtered. Defaults to -1.
        max_det (int): The maximum number of boxes after `postprocess`
        classes (List[str], optional): Category information.
        palette (List[tuple], optional): Palette information corresponding to 
            the category.
    """
    
    def __init__(self, 
                 backend_model: ModelType, 
                 pipeline: Union[Callable, Sequence[Callable]],
                 conf: Optional[float] = None,
                 nms_cfg: Optional[dict] = None,
                 min_box_size: int = -1,
                 max_det: int = 100,
                 classes: Optional[Union[List[str], str]] = None,
                 palette: Optional[Union[List[tuple], str, tuple]] = None):
        valid_nms_params = {'iou_thres', 'class_agnostic', 'nms_pre'} 
        if nms_cfg is not None:
            assert set(nms_cfg) <= valid_nms_params , \
                f'{set(nms_cfg) - valid_nms_params} is invalid nms params'
        if conf is not None:
            assert 0. < conf < 1.
        self.conf = conf
        self.min_box_size = min_box_size
        self.nms_cfg = nms_cfg
        self.max_det = max_det
        
        super().__init__(backend_model, pipeline)
        if isinstance(classes, str):
            if palette is None:
                palette = classes
            classes = Classes[classes].value
        self.classes = classes
        self.palette = palette
        self.visualizer = UniversalVisualizer()

        self._num_aug_vis = 0
    
    def _init_pipeline(self, pipeline: Union[Callable, Sequence[Callable]]
                       ) -> Tuple[Callable, int]:
        """Initialize the preprocess pipeline.

        Return a pipeline to handle various input data, such as ``str``,
        ``np.ndarray``. And an int to specify the number of the strategies of 
        test-time augmention.

        The returned pipeline will be used to process a single data.
        It will be used in :meth:`preprocess` like this:

        .. code-block:: python
            def preprocess(self, inputs, batch_size, **kwargs):
                ...
                dataset = map(self.pipeline, dataset)
                ...
        
        Returns:
            `dlicv.transforms.Compose`: Pipeline to handle various input data.
            int: Specify the number of the strategies of test-time augmention.
                 when it greater than 1. `1` indicates that test-time 
                 augmention has not been used.
        """
        tta_num_aug = 1
        if isinstance(pipeline, (list, tuple)):
            if isinstance(pipeline[-1], TestTimeAug):
                tta_num_aug = len(pipeline[-1].subroutines)
            elif not isinstance(pipeline[-1], PackImgInputs):
                pipeline = list(pipeline)
                pipeline.append(PackImgInputs(DetDataSample))
            pipeline = Compose(pipeline)
        elif isinstance(pipeline, Compose):
            if isinstance(pipeline.transforms[-1], TestTimeAug):
                tta_num_aug = len(pipeline.transforms[-1].subroutines)
            elif not isinstance(pipeline.transforms[-1], PackImgInputs):
                pipeline.transforms.append(PackImgInputs(DetDataSample))
        else:
            pipeline = Compose([pipeline, PackImgInputs(DetDataSample)])
        return pipeline, tta_num_aug
    
    def update_pipeline(self, pipeline):
        self.pipeline, self.tta_num_aug = self._init_pipeline(pipeline)
    
    @abstractmethod
    def _parse_preds(self, preds: Any, 
                     batch_img_metas: Optional[list] = None) -> Tuple:
        """Parse the predictions from :meth:`forward` into bbox results.

        Args:
            preds (Any): Predictions from :meth:`forward`.
            batch_img_metas (Optional[list]): Each item contains the meta 
                information of each image. Defaults to None.

        Returns:
            bboxes (List[Tensor[N, 4]]): Each item is a Tensor representing 
                all detection boxes of each image, has a shape 
                (num_objects, 4). Expected to be (x1, y1, x2, y2) format in 
                pixel coordinates.
            scores (List[Tensor[N,]]): Each item is a Tensor representing 
                detection scores corresponding to the detection boxes of
                each image, has a shape (num_objects, ).
            labels (List[Tensor[N,]]): Each item is a Tensor representing 
                labels corresponding to the detection boxes of each image, has
                a shape (num_objects, ).
        """
    
    def bbox_postprocess(self,
                         bboxes: Tensor,
                         scores: Tensor,
                         labels: Tensor,
                         img_meta: Optional[dict] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. 
        
        This func provides the default workflow as follows:

        1. Filter out those bbox with socre lower than `conf`.
        2. Recale the bbox to the original image scale accroding to image 
           meta-info.
        3. Filter out those bbox with sizes smaller than `min_box_size`.
        4. Apply non-maximum suppression operations to boxes with `nms_cfg` as 
           the kwargs.
        5. Select the top `max_det` detection boxes with the highest scores.

        Args:
            bboxes (Tensor[N, 4]): Detection boxes, has a shape 
                (num_objects, 4), the last dimension 4 arrange as 
                (x1, y1, x2, y2).
            scores (Tensor): Confidence scores, has a shape (num_object, ).
            labels (Tensor): Labels of bboxes, has a shape (num_object, ).
            img_meta (dict, optional): Image meta info. Defautls to None.

        Returns:
            bboxes (Tensor[N_post, 4]): Detection boxes after post-process, 
                has a shape (num_objects, 4), the last dimension 4 arrange as 
                (x1, y1, x2, y2).
            scores (Tensor[N_post,]): Confidence scores, has a shape 
                (num_object_post, ).
            labels (Tensor[N_post,]): Labels of bboxes, has a shape 
                (num_object_post, ).
        """
        assert len(bboxes) == len(scores) == len(labels)
        # filter low confidence boxes 
        if self.conf is not None:
            idxs = scores > self.conf
            bboxes, labels, scores = bboxes[idxs], labels[idxs], scores[idxs]
        # shift, rescale or flip boxes to original image space.
        padding = img_meta.get('padding')
        if padding is not None:
            left_padding, top_padding = padding[:2]
            bboxes -= bboxes.new_tensor([left_padding, top_padding] * 2)
        scale_factor = img_meta.get('scale_factor')
        if scale_factor is not None:
            scale_factor = [1 / s for s in scale_factor]
            bboxes *= bboxes.new_tensor(scale_factor * 2)
        flip_direction = img_meta.get('flip_direction')
        if flip_direction is not None:
            ori_shape = img_meta['ori_shape']
            bboxes = flip_boxes(boxes=bboxes, img_shape=ori_shape, 
                                direction=flip_direction)
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
        """Process a batch of predictions from :meth:`forward` into bbox 
        results.

        Args:
            preds (Any): Predictions of the backend_model.
            batch_datasamples (List[:obj:`DetDataSample`]): Each item 
                contains the meta information of each image.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - bboxes (Tensor[N_post, 4]): Detection boxes after 
                    post-process, has a shape (num_objects, 4), the last 
                    dimension 4 arrange as (x1, y1, x2, y2).
                - scores (Tensor[N_post,]): Confidence scores, has a shape 
                    (num_object_post, ).
                - labels (Tensor[N,]): Labels of bboxes, has a shape 
                    (num_object_post, ).
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_datasamples
        ]
        parsed_preds = self._parse_preds(preds, batch_img_metas)
        for data_sample, bbox_preds, cls_socres, labels, img_meta in \
                zip(batch_datasamples, *parsed_preds, batch_img_metas):

            bboxes, scores, labels = self.bbox_postprocess(
                bbox_preds, cls_socres, labels, img_meta)
            results = InstanceData()
            results.bboxes = bboxes
            results.scores = scores
            results.labels = labels
            data_sample.pred_instances = results
        return batch_datasamples
    
    def tta_merge_bboxes(self, aug_bboxes: List[Tensor],
                         aug_scores: List[Tensor],
                         img_metas: List[str]) -> Tuple[Tensor, Tensor]:
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,4), where
            4 represent (tl_x, tl_y, br_x, br_y)
            and ``scores`` with shape (n,).
        """
        bboxes = torch.cat(aug_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores
    
    def tta_merge_results(self, aug_samples: SampleList) -> DetDataSample:
        """Merge predictions which come form the different views of one image
        to one prediction.

        Args:
            aug_samples (List[DetDataSample]): List of predictions
            of enhanced data which come form one image.
        Returns:
            DetDataSample: Merged prediction.
        """
        aug_bboxes = []
        aug_scores = []
        aug_labels = []
        img_metas = []
        # TODO: support instance segmentation TTA
        assert aug_samples[0].pred_instances.get('masks', None) is None, \
            'TTA of instance segmentation does not support now.'
        for aug_sample in aug_samples:
            aug_bboxes.append(aug_sample.pred_instances.bboxes)
            aug_scores.append(aug_sample.pred_instances.scores)
            aug_labels.append(aug_sample.pred_instances.labels)
            img_metas.append(aug_sample.metainfo)

        merged_bboxes, merged_scores = self.tta_merge_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_labels = torch.cat(aug_labels, dim=0)

        if merged_bboxes.numel() == 0:
            return aug_samples[0]

        keep_idxs = batched_nms(merged_bboxes, merged_scores, merged_labels, 
                                **self.nms_cfg)
        if self.max_det > -1:
            keep_idxs = keep_idxs[:self.max_det]
        det_bboxes = merged_bboxes[keep_idxs]
        det_scores = merged_scores[keep_idxs]
        det_labels = merged_labels[keep_idxs]

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = det_scores
        results.labels = det_labels
        det_results = aug_samples[0]
        det_results.pred_instances = results
        return det_results

    def visualize(self,
                  images: ImgsType,
                  results: List[DetDataSample],
                  show: bool = False,
                  wait_time: float = 0,
                  show_dir: Optional[str] = None,
                  boxes_line_width = 1,
                  show_label = True,
                  **kwargs) -> Optional[List[np.ndarray]]:
        """Visualize predictions.

        Customize your visualization by overriding this method. visualize
        should return visualization results, which could be np.ndarray or any
        other objects.

        Args:
            images (List[ndarray | Tensor]): Original images to vis.
            results (List[DetDataSample]): Results from :meth:`postprocess`.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). 0 is the special
                value that means "forever". Defaults to 0.
            show_dir (str | None): If not None, save the visualization
                results in the specified directory. Defaults to None.

        Returns:
            List[np.ndarray]: Visualization results.
        """
        if not show and show_dir is None:
            return None
        visualizations = []
        for img, result in zip(images, results):
            if isinstance(img, Tensor):
                img = np.ascontiguousarray(
                    img.detach().cpu().numpy().transpose(1, 2, 0))
            img_name = Path(result.img_path).stem if 'img_path' in result \
                else f'{self.num_visualized_imgs:08}'
            if self.tta_num_aug > 1:
                suffix = '' if self._num_aug_vis == self.tta_num_aug else \
                    f'_tta{self._num_aug_vis:02}'
                img_name += suffix
                self._num_aug_vis = (self._num_aug_vis + 1) % (
                    self.tta_num_aug + 1)
            img_name += '.jpg'
            if 'channel_order' in result and result.channel_order != 'rgb':
                img = img[..., ::-1]
            drawn_img = self.visualizer.draw_instances(img, 
                                                       result.pred_instances,
                                                       self.classes,
                                                       self.palette,
                                                       boxes_line_width,
                                                       show_label=show_label)
            if show:
                self.visualizer.show(drawn_img, img_name, wait_time=wait_time)
            if show_dir is not None:
                vis_file = osp.join(show_dir, 'vis', img_name)
                imwrite(vis_file, drawn_img[..., ::-1])
            visualizations.append(img)
        return visualizations