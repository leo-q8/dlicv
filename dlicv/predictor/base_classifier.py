import os.path as osp
from typing import List, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from torch import Tensor
import torch.nn.functional as F

from dlicv.ops import imwrite
from dlicv.structures import ClsDataSample
from dlicv.transforms import PackImgInputs, Compose
from dlicv.utils import Classes
from dlicv.visualization import UniversalVisualizer
from .base import BasePredictor, ModelType

SampleList = List[ClsDataSample]
ImgsType = List[Union[np.ndarray, Tensor]]


class BaseClassifier(BasePredictor):
    """Base image classification predictor.
    
    Args:
        backend_model (dict | torch.nn.Module | BackendModel): A `BackendModel` 
            or `torch.nn.Module` to execute the inference. The `dict` param is 
            for initialing the `BackendModel`. 
        pipeline (Callable | Sequence[Callable]): Data preprocess pipeline. A 
            compose of series of data transformation defined in 
            `dlicv.trasnforms`.
        binary_thres (float | None): Threshold for binary classification in the 
            case of only one class. Default: None.
        use_sigmoid (bool): Whether use sigmoid activation for predict. 
            Default: False.
    """

    def __init__(self, 
                 backend_model: ModelType, 
                 pipeline: Sequence[Callable],
                 binary_thres: Optional[float] = None,
                 use_softmax: bool = False,
                 use_sigmoid: bool = False,
                 classes: Optional[Union[List[str], str]] = None):
        if binary_thres is not None:
            assert 0. < binary_thres < 1.
        assert not (use_sigmoid and use_softmax), ('Only one of `use_sigmoid`'
            ' and `use_softmax` can be set')

        if isinstance(pipeline, (list, tuple)):
            if not isinstance(pipeline[-1], PackImgInputs):
                pipeline = list(pipeline)
                pipeline.append(PackImgInputs(ClsDataSample))
        elif isinstance(pipeline, Compose):
            if not isinstance(pipeline.transforms[-1], PackImgInputs):
                pipeline.transforms.append(PackImgInputs(ClsDataSample))
        else:
            pipeline = Compose([pipeline, PackImgInputs(ClsDataSample)])
        super().__init__(backend_model, pipeline)

        self.binary_thres = binary_thres
        self.use_softmax = use_softmax
        self.use_sigmoid = use_sigmoid
        if isinstance(classes, str):
            classes = Classes[classes].value
        self.classes = classes
        self.visualizer = UniversalVisualizer()
    
    def postprocess(self,
                    cls_preds: Tensor, 
                    batch_datasamples: SampleList, 
                    **kwargs) -> SampleList:
        """Process a batch of predictions from :meth:`forward` into 
        `ClsDataSample`.

        Args:
            cls_preds (Tensor): Batched predicted tensor of the backend_model, 
                with shape (batch, num_classes).
            batch_datasamples (List[:obj:`SegDataSample`]): Each item 
                contains the meta information of each image.

        Returns:
            list[:obj:`ClsDataSample`]: A list of data samples which contains 
                the predicted results. Each ClsDataSample usually contain:

            - ``pred_score``: Prediction of semantic segmentation.
            - ``pred_label``: Predicted probs of semantic
                segmentation after normalization by an activate func.
        """
        B, C = cls_preds.shape
        if self.use_softmax:
            pred_scores = F.softmax(cls_preds, dim=1) 
        elif self.use_sigmoid:
            pred_scores = F.sigmoid(cls_preds)
        else:
            pred_scores = cls_preds
        
        if C > 1: 
            pred_labels = pred_scores.argmax(dim=1, keepdim=True)
        else:
            pred_labels = (pred_scores > self.binary_thres).to(pred_scores)
         
        for data_sample, pred_score, pred_label in zip(
            batch_datasamples, pred_scores, pred_labels):

            data_sample.pred_score = pred_score
            data_sample.pred_label = pred_label
            if self.classes is not None:
                data_sample.pred_class = self.classes[int(pred_label.item())]

        return batch_datasamples
    
    def visualize(self,
                  images: ImgsType,
                  results: SampleList,
                  show: bool = False,
                  wait_time: float = 0,
                  show_dir: Optional[str] = None,
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
            drawn_img = self.visualizer.draw_cls(img, result,
                                                 classes=self.classes)
            if show:
                self.visualizer.show(drawn_img, img_name, wait_time=wait_time)
            if show_dir is not None:
                vis_file = osp.join(show_dir, 'vis', img_name)
                imwrite(vis_file, drawn_img[..., ::-1])
            visualizations.append(img)
        return visualizations