from abc import ABCMeta, abstractmethod
from typing import Dict, Union, Sequence, Tuple, Optional, NamedTuple

import torch
from torch import Tensor

Backend_IOType = Union[Tensor, Sequence[Tensor], Dict[str, Tensor]]


class BackendIOSpec(NamedTuple):
    """Backend model Inputs and Outputs Specs."""

    index: int
    name: str
    shape: Optional[Union[Tuple[int], Dict[str, Tuple[int]]]]
    dtype: Optional[torch.dtype]


class BaseBackend(metaclass=ABCMeta):
    """Abstract base class for inference backend. This class is modified from 
    `mmdeploy` https://github.com/open-mmlab/mmdeploy/ blob/main/mmdeploy/backend/base/base_wrapper.py

    Args:
        backend_files (Sequence[str]): Paths to all required backend files(
            e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        input_specs (Sequence[BackendIOSPec] | None): Specs of backend model 
            inputs. Not necessary for certain backend such as 'TorchScript'.
        output_specs (Sequence[BackendIOSPec] | None): Specs of backend model 
            outputs.
    """

    def __init__(self, 
                 backend_files: Sequence[str],
                 input_specs: Optional[Sequence[BackendIOSpec]] = None,
                 output_specs: Optional[Sequence[BackendIOSpec]] = None):
        super().__init__()
        self._backend_files = backend_files
        self._input_specs = None if input_specs is None else \
            sorted(input_specs, key=lambda x: x.index)
        self._output_specs = None if output_specs is None else \
            sorted(output_specs, key=lambda x: x.index)
    
    @property
    def input_specs(self) -> Optional[Sequence[BackendIOSpec]]:
        return self._input_specs

    @property
    def output_specs(self) -> Optional[Sequence[BackendIOSpec]]:
        return self._output_specs
    
    @staticmethod
    def get_backend_file_count() -> int:
        """Return the count of backend file(s)
        Each backend has its own requirement on backend files (e.g., TensorRT
        requires 1 .engine file and ncnn requires 2 files (.param, .bin)). This
        interface allow developers to get the count of these required files.
        Returns:
            int: The count of required backend file(s).
        """
        return 1

    def infer(self, *inputs: Backend_IOType) -> Backend_IOType:
        if isinstance(inputs[0], (dict, tuple, list)):
            assert len(inputs) == 1
            inputs = inputs[0]
        if type(inputs) == dict:
            assert self._input_specs is not None
            inputs = [inputs[i.name] for i in self._input_specs]
        elif isinstance(inputs, Tensor):
            inputs = [inputs]
        if self._input_specs is not None:
            assert len(self._input_specs) == len(inputs)
         
        return self._infer(inputs)
        
    @abstractmethod
    def _infer(self, inputs: Sequence[Tensor]) -> Backend_IOType:
        """Run forward inference.
        Args:
            inputs (Sequence[np.ndarray]): NDarray Sequence of model inputs.
        Returns:
            Dict[str, np.ndarray]: Key-value pairs of model outputs.
        """
        pass

    def output_to_sequence( self, backend_outputs: Backend_IOType
        ) -> Union[Tensor, Sequence[Tensor]]:
        """Convert the output dict of forward() to a tensor list.
        Args:
            output_dict (Dict[str, np.ndarray]): Key-value pairs of model
                outputs.
        Returns:
            List[np.ndarray]: An output value list whose order is determined
                by the ouput_names list.
        """
        if isinstance(backend_outputs, Tensor):
            return backend_outputs
        elif isinstance(backend_outputs, dict):
            assert self._output_specs is not None and len(self._output_specs)
            backend_outputs = [backend_outputs[output_spec.name] 
                   for output_spec in self._output_specs]
        if len(backend_outputs) == 1:
            backend_outputs = backend_outputs[0]
        return backend_outputs
    
    def __repr__(self) -> str:
        indent = ' ' * 4
        repr_str = f'****{self.__class__.__name__}****\n'
        repr_str += 'Backend_File(s):\n'
        for f in self._backend_files:
            repr_str += indent + f + '\n'
        if self._input_specs is not None:
            repr_str += 'Input(s):\n'
            for index, name, shape, dtype in self._input_specs:
                repr_str += f'{indent}#{index}  {name}  {dtype}  {shape}\n'
        if self._output_specs is not None:
            repr_str += 'Output(s):\n'
            for index, name, shape, dtype in self._output_specs:
                repr_str += f'{indent}#{index}  {name}  {dtype}  {shape}\n'
        return repr_str
