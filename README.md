<div align="center">

# DLICV: Deep Learning Inference kit tool for Computer Vision

[![Static Badge](https://img.shields.io/badge/LICENSE-Apach--2.0-brightgreen)](https://github.com/xueqing888/dlicv/blob/master/LICENSE)
![Static Badge](https://img.shields.io/badge/Python-3.7%2B-blue)
![Static Badge](https://img.shields.io/badge/PyTorch-1.8%2B-orange)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>


## Introduction
DLICV is a Python library developed based on PyTorch for deep learning inference in computer vision tasks. It provides a unified interface for deep learning model inference across different hardware platforms and inference backends, abstracting away many usage details such as resource allocation and release, data movement, etc. DLICV abstracts the deep learning inference process for common computer vision tasks into data preprocessing, backend model inference, post-prediction processing, and inference result visualization. These processes are encapsulated in the basic predictor to realize an end-to-end inference process, avoiding the need for repetitive and cumbersome inference scripting. These features enable DLICV to offer a consistent and convenient deep learning model inference experience for different computer vision tasks on various platforms.
## Main Features
### Multipe hardware platforms and inference backends are available
The supported Device-InferenceBackend matrix is presented as following, and more will be compatible.

| Device / <br> Inference Backend | [ONNX Runtime](https://github.com/microsoft/onnxruntime) | [TensorRT](https://github.com/NVIDIA/TensorRT) | [OpenVINO](https://github.com/openvinotoolkit/openvino) | [ncnn](https://github.com/Tencent/ncnn) | [CANN](https://www.hiascend.com/software/cann) | [CoreML](https://github.com/apple/coremltools) |
| :-----------------------------: | :------------------------------------------------------: | :--------------------------------------------: | :-----------------------------------------------------: | :-------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
|           X86_64 CPU            |                            ✅                             |                                                |                            ✅                            |                                         |                                                |                                                |
|             ARM CPU             |                            ✅                             |                                                |                                                         |                    ✅                    |                                                |                                                |
|             RISC-V              |                                                          |                                                |                                                         |                    ✅                    |                                                |                                                |
|           NVIDIA GPU            |                            ✅                             |                       ✅                        |                                                         |                                         |                                                |                                                |
|          NVIDIA Jetson          |                                                          |                       ✅                        |                                                         |                                         |                                                |                                                |
|          Huawei ascend          |                                                          |                                                |                                                         |                                         |                       ✅                        |                                                |
|            Apple M1             |                                                          |                                                |                                                         |                    ✅                    |                                                |                       ✅                        |


### End-to-end inference process
The `BasePredictor` implemented by DLICV offers an end-to-end inference experience, breaking down the deep learning inference process in common computer vision tasks into four core stages: data preprocessing, backend model inference, post-prediction processing, and inference result visualization. By integrating these four stages into a single basic predictor, DLICV eliminates the need for developers to repeatedly write complex and cumbersome inference scripts, thus enhancing development efficiency.
### Image/bounding box processing support both `np.ndarray` and `torch.Tensor`
- [Image processing](): `imresize`, `impad`, `imcrop`, `imrotate`
- [Image transformation](): `LoadImage`, `Resize`, `Pad`, `ImgToTensor`
- [Bounding box processing](): `clip_boxes`, `resize_boxes`, `box_iou`, `batched_nms`

## Installation
Install DLICV and its basic dependencies:
```bash
pip install git+https://github.com/xueqing888/dlicv.git
```
<details open>
<summary>Install the corresponding inference backend for multi-platform inference</summary>

|    NAME     | INSTALLATION                                                     |
| :---------: | :----------------------------------------------------------- |
| ONNXRuntime | [ONNX Runtime official docs](https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime) offers two Python packages for ONNX Runtime. Only one of these packages should be installed at a time in any one environment. <br />If your platform has CUDA-enabled GPU hardware, we recommend installing the GPU version package, which encompasses most of the CPU functionality.<br /><pre> `pip install onnxruntime-gpu`</pre>Use the CPU package if you are running on Arm CPUs and/or macOS.<br /><pre>`pip install onnxruntime`</pre> |
|  TensorRT   | First, ensure that your platform has the appropriate CUDA version of GPU drivers installed, which can be checked using the `nvidia-smi` command.<br />Then, you can install TensorRT by using the precompiled Python package provided by the [TensorRT repository](https://github.com/NVIDIA/TensorRT?tab=readme-ov-file#prebuilt-tensorrt-python-package)<br /><pre>`pip install tensorrt`</pre> |
|  OpenVINO   | Install  [OpenVINO](https://docs.openvino.ai/2021.4/get_started.html) package<br /><pre>`pip install openvino-dev`</pre> |
|    ncnn     | 1. Download and build ncnn according to its <a href="https://github.com/Tencent/ncnn/wiki/how-to-build">wiki</a>. Make sure to enable <code>-DNCNN_PYTHON=ON</code> in your build command.<br/>2. Export ncnn's root path to environment variable<br/><pre>`cd ncnn`<br />`export NCNN_DIR=$(pwd)`</pre>3. Install pyncnn<br><pre>`cd ${NCNN_DIR}/python`<br/>`pip install -e .`</pre> |
|   Ascend    | 1.Install CANN follow [official guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/60RC1alpha02/softwareinstall/instg/atlasdeploy_03_0002.html).<br/>2. Setup environment<br/>   <pre>`export ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit/latest"`</pre> |

</details>

## Get started

<details open>
<summary>Backend model inference</summary>

The `BackendModel` implemented in DLICV supports inference for multiple backend models. It's straightforward to use: simply pass the relevant backend model file, device type (optional), and other parameters to construct a callable **backend-model** object. You can then perform inference and obtain the results by passing `torch.Tensor` data.

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

<details open>
<summary>Perform end-to-end inference for image classification tasks with <code>BaseClassifier</code>.</summary>

Let's illustrate the usage of `BaseClassifier` with an example of [ResNet18](https://pytorch.org/vision/stable/models/resnet.html#resnet) inference.
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
After successfully running the above code, a directory named `vis` will be created in the current working directory. In this directory, there will be a visualization result image named `dog.jpg` as shown below.

<div align="center">
<img src="figures/dog.jpg" height = 400>
<p></p>
</div>

</details>

<details open>
<summary>Perform end-to-end inference for objectj detection tasks with <code>BaseDetector</code>.</summary>

As an example, let's illustrate the usage of `BaseDetector` with object detection model [YOLOv8](https://github.com/ultralytics/ultralytics). You can refer to the official [model export tutorial](https://docs.ultralytics.com/modes/export) to obtain the backend model you need. Here, we'll demonstrate inference with the onnx model of `yolov8n`

```python
import urllib.resuest

import torch
from dlicv import BackendModel, BaseClassifier
from dlicv.transform import *

# Download an example image from the ultralytics website
url, filename = ("https://ultralytics.com/images/bus.jpg", "bus.jpg")
urllib.request.urlretrieve(url, filename)

# Build BackendModel.
backend_model_file = '/path/to/onnx-model/yolov8n.onnx'
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

After successfully running the above code, a directory named `vis` will be created in the current working directory. In this directory, there will be a visualization result image named `bus.jpg` as shown below.

<div align="center">
<img src="figures/bus.jpg" height = 400>
<p></p>
</div>

</details>

<details open>
<summary>Perform end-to-end inference for semantic segmentation tasks with <code>BaseSegmentor</code></summary>

Let's illustrate the usage of BaseSegmentor with an example of inference using the semantic segmentation model [DeepLabV3](https://pytorch.org/vision/stable/models/deeplabv3.html#deeplabv3).

```python
import urllib.request
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

from dlicv.predictor import BaseSegmentor
from dlicv.transforms import *

# Download an example image from the pytorch website
url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
urllib.request.urlretrieve(url, filename)

# Build DeepLabv3 with pretrained weights from torchvison.
model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights)
model.eval().cuda()

# Build data pipeline for image preprocessing with `dlicv.transforms`
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]
data_pipeline = Compose([
   LoadImage(channel_order='rgb', to_tensor=True, device='cuda'),
   Normalize(mean=MEAN, std=STD),
])

# Build segmentor by subclassing `BaseSegmentor`, and rewrite the 
# method `postprocess`
class DeepLabv3(BaseSegmentor):
    def postprocess(self, preds, *args, **kwargs):
        pred_seg_maps = preds['out']
        return super().postprocess(pred_seg_maps, *args, ** kwargs)

segmentor = DeepLabv3(model, data_pipeline, classes='voc_seg')
res = segmentor(filename, show_dir='./')
```

After successfully running the above code, a directory named `vis` will be created in the current working directory. In this directory, there will be a visualization result image named `deeplab1.jpg` as shown below.

<div align="center">
<img src="figures/deeplab1.png" height = 400>
<p></p>
</div>

</details>

## License
This project is released under the [Apache 2.0 license](LICENSE).
## Acknowledgement
- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
## Citation
If you find this project useful in your research, please consider citing:

```BibTeX
@misc{=dlicv,
    title={Deep Learning Inference kit tool for Computer Vision},
    author={Wang, Xueqing},
    howpublished = {\url{https://github.com/xueqing888/dlicv.git}},
    year={2024}
}
```
