from abc import ABCMeta, abstractmethod
from typing import Any, List, Callable, Union, Sequence

import numpy as np
import torch
import torch.nn as nn

from dlicv.transforms import Compose
from dlicv.structures import BaseDataElement
from ..backend import BackendModel

InputType = Union[np.ndarray, torch.Tensor]
InputsType = Union[InputType, Sequence[InputType]]
ModelType = Union[dict, nn.Module, BackendModel]


class BasePredictor(metaclass=ABCMeta):
    def __init__(self, 
                 backend_model: ModelType,
                 pipeline: Union[Callable, Sequence[Callable]]):
        if isinstance(backend_model, dict):
            backend_model = BackendModel(**backend_model)
        self.backend_model = backend_model
        if isinstance(pipeline, (list, dict)):
            pipeline = Compose(pipeline)
        self.pipeline = pipeline

    @torch.no_grad()
    def __call__(self, 
                 inputs: InputsType, 
                 preporcess_kwargs: dict = dict(),
                 postprocess_kwargs: dict = dict()):

        data = self.preprocess(inputs, **preporcess_kwargs)
        
        preds = self.backend_model(data['inputs'])

        results = self.postprocess(preds, 
                                   data['data_samples'],
                                   **postprocess_kwargs)
        return results
    
    def collect_batch(self, data_batch: Sequence[dict]) -> dict:
        data = {key: [d[key] for d in data_batch]
                for key in data_batch[0]}
        data['inputs'] = torch.stack(data['inputs'])
        return data
    
    def preprocess(self, inputs: InputsType, **kwargs) -> dict:
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        elif isinstance(inputs, torch.Tensor):
            if inputs.ndim == 3:
                inputs = [inputs]
            else:
                assert inputs.ndim == 4, 'Required 3-Dim(C, H, W) or ' \
                    f'4-Dim(B, C, H, W) tensor image'
        data_batch = [self.pipeline({'inputs': input}) for input in inputs] 
        return self.collect_batch(data_batch, **kwargs)

    @abstractmethod
    def postprocess(self, 
                    preds: Any, 
                    batch_datasamples: List[BaseDataElement], 
                    **kwargs):
        pass
    
