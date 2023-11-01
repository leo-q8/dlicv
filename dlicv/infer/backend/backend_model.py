import logging
from typing import Union, Optional, Sequence, Callable

from pathlib import Path
from torch import Tensor

from dlicv.utils import Backend
from dlicv.utils.logging import get_root_logger
from .base import get_backend_manager

format_backends = {
    'engine': Backend.TENSORRT,
    'trt': Backend.TENSORRT,
    'om': Backend.ASCEND,
    'onnx': Backend.ONNXRUNTIME,
    'torchscript': Backend.TORCHSCRIPT,
    'pt': Backend.TORCHSCRIPT,
}


class BackendModel:
    """A backend model wraps the details to initialize and run a backend
    engine."""

    def __init__(self,
                 backend_files: Union[str, Sequence[str]],
                 backend: Optional[Backend] = None,
                 device_type: str = 'cpu',
                 device_id: int = 0,
                 logger: Optional[logging.Logger] = None,
                 preprocessor: Optional[Callable] = None,
                 postprocessor: Optional[Callable] = None,
                 input_names: Optional[Sequence[str]] = None,
                 output_names: Optional[Sequence[str]] = None,
                 force_cast: bool = False,
                 **kwargs):
        """The default methods to build backend wrappers.
        Args:
            backend (Backend): The backend enum type.
            beckend_files (Sequence[str]): Paths to all required backend files(
                e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string specifying device type.
            input_names (Sequence[str] | None): Names of model inputs in
                order. Defaults to `None`.
            output_names (Sequence[str] | None): Names of model outputs in
                order. Defaults to `None` and the wrapper will load the output
                names from the model.
        """
        self.logger = get_root_logger() if logger is None else logger
        if backend is None:
            if isinstance(backend_files, str):
                file_format = backend_files.split('.')[-1]
                if file_format not in format_backends:
                    raise TypeError(
                        f'No defualt associated backend for `{file_format}` '
                        f'format backend-model file! Please specify '
                        f'`backend` param manually.')
                backend = format_backends[file_format]
                self.logger.info(
                    f'Automatically specify the `{backend}` backend for '
                    f'`{Path(backend_files).name}` backend-file.')
            elif isinstance(backend_files, Sequence):
                raise TypeError(
                    f'Please specify `backend` param manually for `List` ' 
                    f'type `backend_files` input.')
            else:
                raise ValueError(
                    f'Only support `str` or `Sequence[str]` type for '
                    f'param `backend_files`, but got {type(backend_files)}')
        backend_mgr = get_backend_manager(backend.value)
        if backend_mgr is None:
            raise NotImplementedError(
                f'Unsupported backend type: {backend.value}')
        if isinstance(backend_files, str):
            backend_files = [backend_files]
        self.backend = backend_mgr.build_backend(backend_files, device_type, 
                                                 device_id, input_names, 
                                                 output_names, **kwargs)
        self.backend_mgr = backend_mgr
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.force_cast = force_cast
        backend_file = backend_files if isinstance(backend_files, str) else \
                                         backend_files[0]
        self._name = Path(backend_file).name
        self._no_more_warning = False

    def destroy(self):
        if hasattr(self, 'backend') and hasattr(self.backend, 'destroy'):
            self.backend.destroy()
    
    def __call__(self, inputs: Union[Tensor, Sequence[Tensor]], 
                 preprocessor_kargs : dict = dict(),
                 postprocessor_kargs : dict = dict()):
        pre_inputs = inputs if not self.with_preprocessor else \
            self.preprocessor(inputs, **preprocessor_kargs)
        if isinstance(pre_inputs, Tensor): 
            pre_inputs = [pre_inputs]
        if self.force_cast and self.backend.input_specs is not None:
            inputs = []
            for input, input_spec in zip(pre_inputs, self.backend.input_specs):
                if input.dtype is not None and input.dtype != input_spec.dtype:
                    if not self._no_more_warning:
                        self.logger.warning(
                            f"Backend model `{self._name}` requires "
                            f"{input_spec.dtype} for input "
                            f"`{input_spec.name}`. Cast {input.dtype} input "
                            f"to {input_spec.dtype}")
                        self._no_more_warning = True
                    input = input.to(input_spec.dtype)
                inputs.append(input)
        else:
            inputs = pre_inputs
        outputs = self.backend.infer(inputs)
        outputs = self.backend.output_to_sequence(outputs)
        if self.with_postprocessor:
            outputs = self.postprocessor(outputs, **postprocessor_kargs)
        return outputs

    @property    
    def with_preprocessor(self) -> bool:
        """bool: whether the Model has a preprocessor"""
        return hasattr(self, 'preprocessor') and self.preprocessor is not None

    @property    
    def with_postprocessor(self) -> bool:
        """bool: whether the Model has a postprocessor"""
        return hasattr(self, 'postprocessor') and \
               self.postprocessor is not None
    
    def __repr__(self) -> str:
        repr_str = "****Env info****\n"
        env_info = self.backend_mgr.check_env()
        repr_str += env_info + '\n'
        repr_str += repr(self.backend)
        return repr_str
