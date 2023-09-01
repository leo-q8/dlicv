from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Dict, Union, Sequence, Optional

import numpy as np

Backend_IOType = Union[np.ndarray, Sequence[np.ndarray], Dict[str, np.ndarray]]

#TODO a cython bug, do not suport NamedTuple object cythonized
# class BackendIOSpec(NamedTuple):
#     name: str
#     index: int
#     shape: Optional[Union[Tuple[int], Dict[str, Tuple[int]]]]
#     dtype: Optional[np.dtype]

BackendIOSpec = namedtuple('BackendIOSpec', "name, index, shape, dtype")


class BaseBackend(metaclass=ABCMeta):
    """Abstract base class for backend.
    Args:
        output_names (Sequence[str]): Names to model outputs in order, which is
        useful when converting the output dict to a ordered list or converting
        the output ordered list to a key-value dict.
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

    def infer(self, inputs: Backend_IOType) -> Backend_IOType:
        if type(inputs) == dict:
            assert self._input_specs is not None and len(self._input_specs)
            inputs = [inputs[i.name] for i in self._input_specs]
        elif isinstance(inputs, np.ndarray):
            inputs = [inputs]
        return self._infer(inputs)
        
    @abstractmethod
    def _infer(self, inputs: Sequence[np.ndarray]) -> Backend_IOType:
        """Run forward inference.
        Args:
            inputs (Sequence[np.ndarray]): NDarray Sequence of model inputs.
        Returns:
            Dict[str, np.ndarray]: Key-value pairs of model outputs.
        """
        pass

    def output_to_sequence(
            self, backend_outputs: Backend_IOType) -> Backend_IOType:
        """Convert the output dict of forward() to a tensor list.
        Args:
            output_dict (Dict[str, np.ndarray]): Key-value pairs of model
                outputs.
        Returns:
            List[np.ndarray]: An output value list whose order is determined
                by the ouput_names list.
        """
        if isinstance(backend_outputs, np.ndarray):
            return backend_outputs
        elif isinstance(backend_outputs, dict):
            assert self._output_specs is not None and len(self._output_specs)
            backend_outputs = [backend_outputs[output_spec.name] 
                   for output_spec in self._output_specs]
        if len(backend_outputs) == 1:
            backend_outputs = backend_outputs[0]
        return backend_outputs
    
    def __repr__(self) -> str:
        repr_str = f'****{self.__class__.__name__}****\n'
        repr_str += 'Backend_File(s):\n'
        for f in self._backend_files:
            repr_str += '\t' + f + '\n'
        if self._input_specs is not None:
            repr_str += 'Input(s):\n'
            for name, index, shape, dtype in self._input_specs:
                repr_str += f'\t#{index}  {name}  {dtype}  {shape}\n'
        if self._output_specs is not None:
            repr_str += 'Output(s):\n'
            for name, index, shape, dtype in self._output_specs:
                repr_str += f'\t#{index}  {name}  {dtype}  {shape}\n'
        return repr_str
