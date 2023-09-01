from abc import ABCMeta, abstractmethod
from typing import Any, List, Callable, Union, Sequence

import numpy as np

from dlcv.transforms import Compose
from dlcv.structures import BaseDataElement
from ..backend import BackendModel

InputType = np.ndarray
InputsType = Union[InputType, Sequence[InputType]]


class BasePredictor(metaclass=ABCMeta):
    def __init__(self, 
                 backend_model: Union[dict, BackendModel],
                 pipeline: Sequence[Callable]):
        if isinstance(backend_model, dict):
            backend_model = BackendModel(**backend_model)
        self.backend_model = backend_model
        self.pipeline = Compose(pipeline)

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
        data['inputs'] = np.stack(data['inputs'])
        return data
    
    def preprocess(self, inputs: InputsType, **kwargs) -> dict:
        data_batch = [self.pipeline(input) for input in inputs] 
        return self.collect_batch(data_batch, **kwargs)

    @abstractmethod
    def postprocess(self, 
                    preds: Any, 
                    batch_datasamples: List[BaseDataElement], 
                    **kwargs):
        pass
    
