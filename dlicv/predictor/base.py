from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
import glob
import inspect
import os
from typing import (Any, Callable, Dict, List, Union, Sequence, Tuple, 
                    Iterable, Optional)

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from rich.progress import track

from dlicv.transforms import Compose
from dlicv.structures import BaseDataElement
from ..backend import BackendModel

InputType = Union[str, np.ndarray, torch.Tensor]
InputsType = Union[InputType, Sequence[InputType]]
ModelType = Union[dict, nn.Module, BackendModel]

IMG_EXTENSIONS = ('jpg', 'jpeg', 'png', 'ppm', 'bmp', 
                  'pgm', 'tif', 'tiff', 'webp')


class PredictorMeta(ABCMeta):
    """Automatically capture keyword parameters for predictor.

    This MetaClass firstly captures the keyword parameters set of 
    :meth:`preprocess`, :meth:`forward`, :meth:`postprocess` and 
    :meth:`visualize`. And save as the class attribute ``preprocess_kwargs``,
    ``forward_kwargs``, ``postprocess_kwargs`` and ``visualize_kwargs`` 
    respectively. Next, save the duplicate keyword parameters and it's function 
    names to ``conflict_kwargs``, and save the set of all parameters in 
    ``all_kwargs``
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        expt_args = {'self', 'kwargs'}
        self.preprocess_kwargs = set(inspect.signature(
            self.preprocess).parameters.keys()) - expt_args
        self.forward_kwargs = set(inspect.signature(
            self.forward).parameters.keys()) - expt_args
        self.postprocess_kwargs = set(inspect.signature(
            self.postprocess).parameters.keys()) - expt_args
        self.visualize_kwargs = set(inspect.signature(
            self.visualize).parameters.keys()) - expt_args

        self.all_kwargs = self.preprocess_kwargs | self.forward_kwargs | \
            self.visualize_kwargs | self.postprocess_kwargs

        conflict_kwargs = defaultdict(set)
        KwargsTuple = namedtuple('KwargsTuple', ['func_name', 'kwargs'])
        kwargs_list = [
            KwargsTuple('preprocess', self.preprocess_kwargs),
            KwargsTuple('forward', self.forward_kwargs),
            KwargsTuple('postprocess', self.postprocess_kwargs),
            KwargsTuple('visualize', self.visualize_kwargs)
        ]
        for i, kwt1 in enumerate(kwargs_list):
            for kwt2 in kwargs_list[i+1:]:
                for kw in kwt1.kwargs & kwt2.kwargs:
                    conflict_kwargs[kw].update((kwt1.func_name, 
                                                kwt2.func_name))
        self.conflict_kwargs = conflict_kwargs


class BasePredictor(metaclass=PredictorMeta):
    """Base predictor for downstream computer vision tasks. 

    The BasePredictor provides the standard workflow for inference as follows:

    1. Preprocess the input data by :meth:`preprocess`.
    2. Forward the data to the model by :meth:`forward`. The type of 
       `backend_model` is `torch.nn.Module` or `dlicv.BackendModel` 
       and it's :meth:``__call__`` will be called by default.
    3. Postprocess the backend_model's outputs and return the results 
       by :meth:`postprocess`.
    4. Visualize the results by :meth:`visualize`.

    When we call the subclasses inherited from BasePredictor (not overriding
    ``__call__``), the workflow will be executed in order.

    Subclasses inherited from ``BasePredictor`` should implement
    :meth:`postprocess` and :meth:`visualize`

    This class is modified from `mmengine` 
    https://github.com/open-mmlab/mmengine/blob/main/mmengine/infer/infer.py

    Args:
        backend_model (dict | torch.nn.Module | BackendModel): A `BackendModel` 
            or `torch.nn.Module` to execute the inference. The `dict` param is 
            for initialing the `BackendModel`. 
        pipeline (Callable | Sequence[Callable]): Data preprocess pipeline. A 
            compose of series of data transformation defined in 
            `dlicv.trasnforms`.
    """
    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = set()
    postprocess_kwargs: set = set()
    conflict_kwargs :defaultdict = defaultdict(set)
    all_kwargs:set = set()

    def __init__(self, 
                 backend_model: ModelType,
                 pipeline: Union[Callable, Sequence[Callable]]) -> None:
        if isinstance(backend_model, dict):
            backend_model = BackendModel(**backend_model)
        self.backend_model = backend_model
        if isinstance(pipeline, (list, tuple)):
            pipeline = Compose(pipeline)
        self.pipeline = pipeline
        # A global counter tracking the number of images processed, for
        # naming of the output images
        self.num_visualized_imgs = 0

    def __call__(self, 
                 inputs: InputsType, 
                 batch_size: Optional[int] = None,
                 show: bool = False,
                 wait_time: float = 0,
                 show_dir: Optional[Union[str, Path]] = None,
                 show_progress: bool = False,
                 return_vis: bool = False,
                 **kwargs) -> dict:
        """Call the predictor.  

        The ``BasePredictor`` assumes the data processed by ``pipeline`` 
        is a dict containing keys of 'ori_imgs', 'inputs' and 'data_samples', 
        representing the original inputs, the model-feedable input data and a
        `dlicv.structure.BaseDataElement` instance which warps some meta-info 
        useful for :meth:`postprocess`.

        Args:
            inputs (InputsType): Inputs for the predictor.
            batch_size (int): Batch size. Defaults to None.
            show (bool): Whether to display the visualization results in a
                popup window. Defaults to False.
            wait_time (float): The interval of show (s). 0 is the special
                value that means "forever". Defaults to 0.
            show_dir (str, optional): If not None, save the visualization
                results in the specified directory. Defaults to None.
            show_progress (bool): Control whether to display the progress bar 
                during the inference process. Defaults to False.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)
        # Preprocess the user inputs into a model-feedable fromat.
        inputs = self.preprocess(inputs, batch_size, **preprocess_kwargs)

        results, visualizations = [], [] 
        for data in (track(inputs, description='Predict') 
                     if show_progress else inputs):
            # Predict the data with the `backend_model`.
            preds = self.forward(data['inputs'], **forward_kwargs)
            # Postprocess the backend_model's outputs.
            batch_results = self.postprocess(preds, 
                                             data['data_samples'],
                                             **postprocess_kwargs)
            results.extend(batch_results)
            # Visualize the results.
            batch_visualizations = self.visualize(data['ori_imgs'], 
                                                  batch_results, 
                                                  show=show, 
                                                  wait_time=wait_time,
                                                  show_dir=show_dir,
                                                  **visualize_kwargs)
            if batch_visualizations is not None:
                visualizations.extend(visualizations)   

        return {'results': results, 'visualizations': visualizations} if \
            return_vis else {'results': results}

    def _inputs_to_list(self, 
                        inputs: InputsType,
                        batch_size: Optional[int] = None) -> Tuple[list, int]:
        """Preprocess the inputs to a list and derive batch size based on input
        type and `batch_size` specified by user.

        - list or tuple: return inputs
        - str:
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              is a path to file.
        - other: return a list with one item.

        Args:
            inputs (str | array | tensor | list): Inputs for the predictor.

        Returns:
            list: List of input for the :meth:`preprocess`.
            int: Batch size for infer.
        """
        if isinstance(inputs, str):
            path = str(Path(inputs).absolute())
            if os.path.isdir(path):
                batch_size = 1 if batch_size is None else batch_size
                inputs = sorted(
                    filter(lambda f: f.split('.')[-1] in IMG_EXTENSIONS,
                    glob.glob(os.path.join(path, '*.*'))))
            elif os.path.isfile(path):
                batch_size = 1
        elif isinstance(inputs, torch.Tensor) and inputs.ndim == 4:
            batch_size = len(inputs) if batch_size is None else batch_size
            inputs = [input for input in inputs]
            
        if not isinstance(inputs, (list, tuple)):
            batch_size = 1 if batch_size is None else batch_size
            inputs = [inputs]

        inputs = [{'img_path' if isinstance(input, str) else 'ori_img': input}
                  for input in inputs]
                   
        return inputs, batch_size

    def _collect_batch(self, data_batch: Sequence[dict]) -> dict:
        """Convert list of data sampled from ``pipeline`` into a batch of data.

        Args:
            data_batch (Sequence[dict]): List of dict data from ``pipeline``
        
        Returns:
            dict: Batched data with dict type.
        """
        data = {key: [d[key] for d in data_batch]
                for key in data_batch[0]}
        data['inputs'] = torch.stack(data['inputs'])
        return data
    
    def _get_chunk_data(self, inputs: Iterable, chunk_size: int):
        """Get batch data from dataset.

        Args:
            inputs (Iterable): An iterable dataset.
            chunk_size (int): Equivalent to batch size.

        Yields:
            list: batch data.
        """
        inputs_iter = iter(inputs)
        while True:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    processed_data = next(inputs_iter)
                    chunk_data.append(processed_data)
                yield chunk_data
            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break
    
    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
        """Process the inputs into a model-feedable format.

        Customize your preprocess by overriding this method. Preprocess should
        return an dict object with keys of `inputs` and `data_samples.  The 
        'inputs' is the stack of batch images tensor.  And the 'data_samples' 
        contains the meta information of input images, which is useful for the 
        ``postprocess``.

        ``BasePredictor.preprocess`` will return an iterable chunked data,
        which will be used in __call__ like this:

        .. code-block:: python

            def __call__(self, inputs, batch_size=1, **kwargs):
                chunked_data = self.preprocess(inputs, batch_size, **kwargs)
                for batch in chunked_data:
                    preds = self.forward(batch, **kwargs)

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1

        Returns:
            Any: Data processed by the ``pipeline`` and ``_collect_batch``.
        """
        list_inputs, batch_size = self._inputs_to_list(inputs, batch_size)
        chunked_data = self._get_chunk_data(
            map(self.pipeline, list_inputs), batch_size)
        yield from map(self._collect_batch, chunked_data)

    @torch.no_grad()
    def forward(self, inputs: Union[dict, tuple], **kwargs) -> Any:
        """Feed the inputs to the model."""
        return self.backend_model(inputs)

    @abstractmethod
    def postprocess(self, 
                    preds: Any, 
                    batch_datasamples: List[BaseDataElement], 
                    **kwargs) -> List[BaseDataElement]:
        """Process the predictions results from ``forward``.

        Customize your postprocess by overriding this method. 

        Args:
            preds (Any): Predictions of the `backend_model`.
            batch_datasamples (List[:obj:`BaseDataElement`]): Each item 
                contains the meta information of each image.

        Returns:
            list: Prediction results.
        """
   
    def visualize(self,
                  inputs: InputsType,
                  results: List[BaseDataElement],
                  show: bool = False,
                  wait_time: float = 0,
                  show_dir: Optional[Union[str, Path]] = None,
                  **kwargs) -> Optional[List[np.ndarray]]:
        """Visualize predictions.

        Customize your visualization by overriding this method. visualize
        should return visualization results, which could be np.ndarray or any
        other objects.

        Args:
            inputs (InputsType): Inputs given by user.
            results (Any): Results from :meth:`postprocess`.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). 0 is the special
                value that means "forever". Defaults to 0.
            show_dir (str | Path | None): If not None, save the visualization
                results in the specified directory. Defaults to None.

        Returns:
            List[np.ndarray]: Visualization results.
        """
        pass

    def _dispatch_kwargs(self, **kwargs) -> Tuple[Dict, Dict, Dict, Dict]:
        """Dispatch kwargs to preprocess(), forward(), visualize() and
        postprocess() according to the actual demands.

        Returns:
            Tuple[Dict, Dict, Dict, Dict]: kwargs passed to preprocess,
            forward, visualize and postprocess respectively.
        """
        union_kwargs = self.all_kwargs | set(kwargs.keys())
        if union_kwargs != self.all_kwargs:
            unknown_kwargs = union_kwargs - self.all_kwargs
            raise ValueError(
                f'unknown argument {unknown_kwargs} for `preprocess`, '
                '`forward`, `visualize` and `postprocess`')
            
        # Ensure each argument only matches one function
        conflict_kwargs = set(self.conflict_kwargs) & set(kwargs)
        if conflict_kwargs:
            errmsgs = [
                f'conflict argument `{kw}` for {self.conflict_kwargs[kw]}'
                for kw in conflict_kwargs
            ]
            raise ValueError('\n'.join(errmsgs))

        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}
        visualize_kwargs = {}

        for key, value in kwargs.items():
            if key in self.preprocess_kwargs:
                preprocess_kwargs[key] = value
            elif key in self.forward_kwargs:
                forward_kwargs[key] = value
            elif key in self.postprocess_kwargs:
                postprocess_kwargs[key] = value
            else:
                visualize_kwargs[key] = value

        return (
            preprocess_kwargs,
            forward_kwargs,
            postprocess_kwargs,
            visualize_kwargs,
        )