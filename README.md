## 简介
DLICV是一个基于PyTorch开发，用于在计算机视觉任务中进行深度学习推理的python库。针对不同的硬件平台和推理后端，它供了深度学习模型推理的统一接口，屏蔽了不同推理后端的诸多使用细节诸如资源申请释放、数据搬运等。DLICV将常见计算机视觉基础任务的深度学习推理过程抽象为数据前处理、后端模型推理、预测结果后处理和推理结果可视化，并将上述流程封装在基础预测器中实现端到端的推理过程，避免重复编写繁琐的推理脚本。上述特性使得DLICV可以在不同平台上针对不同任务提供一致和便捷的深度学习模型推理体验。
## 主要特性(Main Features)
### 支持多种硬件平台和推理后端
支持的硬件平台和推理后端如下表所示
The supported Device-InferenceBackend matrix is presented as following,
| Device / Inference Backend | ONNX Runtime | OpenVINO | TensorRT | CANN | CoreML | ncnn |
| :------------------------: | :----------: | :------: | :------: | :--: | :----: | :--: |
### 端到端的推理流程
DLICV实现的`BasePredictor`提供了端到端的推理体验，它将常见的计算机视觉基础任务中的深度学习推理过程分解为四个核心环节：数据预处理、后端模型推理、预测结果后处理和推理结果可视化。通过将这四个环节整合到一个基础预测器中，DLICV避免了开发者需要重复编写复杂且繁琐的推理脚本，从而提高了开发效率和便利性。
### 提供同时支持`np.ndarry`和`torch.Tenosr`的多种常用的图像、边界框处理函数
- [图像处理](): `imread`, `imwrite`, `imresize`, `impad`, `imcrop`, `imrotate`
- [图像变换](): `LoadImage`, `Resize`, `Pad`, `ImgToTensor`
- [边界框处理](): `clip_boxes`, `resize_boxes`, `box_iou`, `batched_nms`

## 安装(Installation)
安装DLICV和基础依赖包
```bash
pip install git+https://github.com/avril-wang1214/dlicv.git
```
<details>
<summary>为了实现多平台推理，需要安装对应推理后端及所提供的Python SDK</summary>

</details>

## 快速上手(Get started)

<details>
<summary>后端模型推理</summary>

DLICV实现的`BackendModel`支持多种推理后端模型的推理。使用起来也非常简单，传入相应的后端模型文件、设备类型（可选）等参数构建一个可调用**后端模型**对象。传入`torch.Tensor`数据就可进行推理，获取推理结果。

```python
import dlicv
import torch
from dlicv import BackendModel

X = torch.randn(1, 3, 224, 224)

onnx_file = '/path/to/onnx_model.onnx'
onnx_model = BackendModel(onnx_file)
onnx_preds = onnx_model(X, force_cast=True)

trt_file = '/path/to/tensorrt_model.trt'
trt_model = BackendModel(trt_file)
trt_pred = trt_model(X, force_cast=True)
```

</details>

<details>
<summary>使用基础预测器进行端到端推理</summary>

DLIC将深度学习推理的整个过程分解为四个核心环节：数据预处理、后端模型推理、预测结果后处理和推理结果可视化。通过将这四个环节整合到基础预测器`BasePredictor`中，DLICV带来了端到端的推理体验，避免了重复编写复杂且繁琐的推理脚本，提高开发效率。

针对不同的基础计算机视觉任务，DLICV实现了相应的基础预测器：`BaseClassifier`, `BaseDetector`和`BaseSegmentor`分别对图像分类、目标检测和语义分割的深度学习模型进行端到端推理，它们都继承自`BasePredictor`。下面将分别介绍如何使用上述三个基础预测器。

我们首先以[Resnet18](https://pytorch.org/vision/stable/models/resnet.html#resnet)的推理为例介绍`BaseClassifier`的使用

```python
import urllib.request

import dlicv
import torch
from dlicv import BaseClassifier
from dlicv.transform import *
from torchvision.models.resnet import resnet18, ResNet18_Weights

# Download an example image from the pytorch website
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
# Build resnet18 with ImageNet 1k pretrained weights from torchvison.
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval().cuda()

# Build data pipeline for image preprocessing with `dlicv.transforms`
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]
data_pipeline = Compose([
   LoadImage(channel_order='rgb', to_tensor=True, device='cuda'),
   Resize(224),
   Pad(to_square=True, pad_val=114),
   Normalize(mean=MEAN, std=STD),
])

# Build Classifier
classifier = BaseClassifier(model, data_pipeline, classes='imagenet')
res = classifier(filename, show_dir='./') # 
```

我们以目标检测模型[YOLOv8](https://github.com/ultralytics/ultralytics)的推理为例介绍`BaseDetector`的使用
可以参考yolov8官方给的[模型导出教程]()来获取你想要的后端模型

```python
import urllib.resuest

import torch
from dlicv import BackendModel, BaseClassifier
from dlicv.transform import *

# Download an example image from the ultralytics website
url, filename = ("https://ultralytics.com/images/bus.jpg", "bus.jpg")
urllib.request.urlretrieve(url, filename)

# Build BackendModel.
backend_model_file = '/path/to/backend model/file'
backend_model = BackendModel(backend_model_file)

# Build data pipeline for image preprocessing with `dlicv.transforms`
data_pipeline = (
    LoadImage(channel_order='rgb'),
    Resize((640, 640)),
    Normalize(mean=0, std=255),
    ImgToTensor()
)

# Build detector by subclassing `BaseDetector`, and implement the abstract
# method `_parse_preds` to parse the predictions from backend model into 
# bbox results
class YOLOv8(BaseDetector):
    def _parse_preds(self, preds: torch.Tensor, *args, **kargs) -> tuple:
        scores, boxes, labels = [], [], []
        outputs = preds.permute(0, 2, 1)
        for output in outputs:
            classes_scores = output[:, 4:]
            cls_scores, cls_labels = classes_scores.max(-1)
            scores.append(cls_scores)
            labels.append(cls_labels)

            x, y, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
            x1, y1 = x - w / 2, y - h / 2
            x2, y2 = x + w / 2, y + h / 2
            boxes.append(torch.stack([x1, y1, x2, y2], 1))
        return boxes, scores, labels

# Init Detector
detector = YOLOv8(backend_model, 
                  data_pipeline, 
                  conf=0.5,
                  nms_cfg=dict(iou_thres=0.5, class_agnostic=True),
                  classes='coco')
res = detector(filename, show_dir='.') 
```

</details>

## 开源许可证(License)
该项目采用 Apache 2.0 开源许可协证
This project is released under the [Apache 2.0 license](LICENSE).
## 致谢(Acknowledgement)
- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
## 引用(Citation)
如果您在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 DLICV:
If you find this project useful in your research, please consider citing:

```BibTeX
@misc{=dlicv,
    title={OpenMMLab's Model Deployment Toolbox.},
    author={Wang, Xueqing},
    howpublished = {\url{https://github.com/open-mmlab/mmdeploy}},
    year={2024}
}
```

 