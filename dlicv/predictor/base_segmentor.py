import os.path as osp
from typing import List, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from torch import Tensor
import torch.nn.functional as F

from dlicv.ops import imwrite
from dlicv.structures import SegDataSample, PixelData
from dlicv.transforms import Compose, PackImgInputs
from dlicv.utils import Classes
from dlicv.visualization import UniversalVisualizer
from .base import BasePredictor, ModelType

SampleList = List[SegDataSample]
ImgsType = List[Union[np.ndarray, Tensor]]

class BaseSegmentor(BasePredictor):
    """Base Semantic segmentation predictor.
    
    Args:
        backend_model (dict | torch.nn.Module | BackendModel): A `BackendModel` 
            or `torch.nn.Module` to execute the inference. The `dict` param is 
            for initialing the `BackendModel`. 
        pipeline (Callable | Sequence[Callable]): Data preprocess pipeline. A 
            compose of series of data transformation defined in 
            `dlicv.trasnforms`.
        binary_thres (float | None): Threshold for binary segmentation in the 
            case of only one output channel. Default: None.
        align_corners (bool): align_corners argument for resize segmentation 
            map in :meth:`mask_postprocess` Default: False.
        use_sigmoid (bool): Whether use sigmoid activation for predict 
            segmention maps from backend_model. Default: False.
    """

    def __init__(self, 
                 backend_model: ModelType, 
                 pipeline: Sequence[Callable],
                 binary_thres: Optional[float] = None,
                 align_corners: bool = False,
                 use_sigmoid: bool = False,
                 classes: Optional[Union[List[str], str]] = None,
                 palette: Optional[Union[List[tuple], str, tuple]] = None):
        if binary_thres is not None:
            assert 0. < binary_thres < 1.
        self.binary_thres = binary_thres
        self.align_corners = align_corners
        self.use_sigmoid = use_sigmoid

        if isinstance(pipeline, (list, tuple)):
            if not isinstance(pipeline[-1], PackImgInputs):
                pipeline = list(pipeline)
                pipeline.append(PackImgInputs(SegDataSample))
        elif isinstance(pipeline, Compose):
            if not isinstance(pipeline.transforms[-1], PackImgInputs):
                pipeline.transforms.append(PackImgInputs(SegDataSample))
        else:
            pipeline = Compose([pipeline, PackImgInputs(SegDataSample)])
        super().__init__(backend_model, pipeline)

        if isinstance(classes, str):
            if palette is None:
                palette = classes
            classes = Classes[classes].value
        self.classes = classes
        self.palette = palette
        self.visualizer = UniversalVisualizer()
    
    def seg_map_postprocess(self,
                            seg_map: Tensor,
                            img_meta: dict) -> Tuple[Tensor, Tensor]:
        """seg_map post-processing method.

        The seg map would be rescaled to the original image. And would be 
        normalized if `use_sigmoid` is True.

        Args:
            seg_map (Tensor[C, H, W]): Predicted segmention map from 
                :meth:`forward`.
            img_meta (dict, optional): Image meta info.

        Returns:
            seg_prob (Tensor[C, H, W]): Predicted prob map of semantic 
                segmentation.
            sem_seg (Tensor[1, H, W]): Prediction of semantic segmentation.
        """
        C, H, W = seg_map.shape

        # remove padding area.
        padding = img_meta.get('padding')
        if padding is not None:
            left_padding, top_padding, right_padding, bottom_padding = padding
            seg_map = seg_map[..., top_padding: H - bottom_padding,
                                    left_padding: W - right_padding]
        
        # resize as original shape
        scale_factor = img_meta.get('scale_factor')
        if scale_factor is not None:
            ori_shape = img_meta.get('ori_shape')
            seg_map = F.interpolate(
                seg_map[None], ori_shape, mode='bilinear', 
                align_corners=self.align_corners).squeeze(0)
        
        if self.use_sigmoid:
            seg_map = seg_map.sigmoid()
        
        if C > 1: 
            seg_pred = seg_map.argmax(dim=0, keepdim=True)
        else:
            seg_pred = (seg_map > self.binary_thres).to(seg_map)

        return seg_map, seg_pred

    def postprocess(self,
                    pred_seg_maps: Tensor, 
                    batch_datasamples: SampleList, 
                    **kwargs) -> SampleList:
        """Process a batch of predictions from :meth:`forward` into 
        `SegDataSample`.

        Args:
            pred_seg_maps (Tensor): Predicted batched seg maps tensor of the 
                backend_model, with shape (batch, channel, height, width).
            batch_datasamples (List[:obj:`SegDataSample`]): Each item 
                contains the meta information of each image.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the input 
                images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.

            - ``seg_probs``(PixelData): Predicted probs of semantic
                    segmentation after normalization by an activate func.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_datasamples
        ]
        for data_sample, seg_map, img_meta in \
                zip(batch_datasamples, pred_seg_maps, batch_img_metas):

            seg_probs, seg_pred = self.seg_map_postprocess(seg_map, img_meta)
            data_sample.seg_probs = PixelData(**{'data': seg_probs})
            data_sample.pred_sem_seg = PixelData(**{'data': seg_pred})
        return batch_datasamples
    
    def visualize(self,
                  images: ImgsType,
                  results: SampleList,
                  show: bool = False,
                  wait_time: float = 0,
                  show_dir: Optional[str] = None,
                  show_labels: bool = True,
                  **kwargs) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
        """Visualize predictions.

        Customize your visualization by overriding this method. visualize
        should return visualization results, which could be np.ndarray or any
        other objects.

        Args:
            images (List[ndarray | Tensor]): Original images to vis.
            results (Any): Results from :meth:`postprocess`.
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
            img_name = osp.basename(result.img_path) if 'img_path' in result \
                else f'{self.num_visualized_imgs:08}.jpg'
            if 'channel_order' in result and result.channel_order != 'rgb':
                img = img[..., ::-1]
            drawn_img = self.visualizer.draw_sem_seg(img, 
                                                     result.pred_sem_seg,
                                                     classes=self.classes,
                                                     palette=self.palette,
                                                     show_labels=show_labels)
            if show:
                self.visualizer.show(drawn_img, img_name, wait_time=wait_time)
            if show_dir is not None:
                vis_file = osp.join(show_dir, 'vis', img_name)
                imwrite(vis_file, drawn_img[..., ::-1])
            visualizations.append(img)
        return visualizations