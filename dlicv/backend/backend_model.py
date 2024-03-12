from typing import Union, Optional, Sequence

from pathlib import Path
from torch import Tensor

from dlicv.utils import Backend
from dlicv.utils import get_root_logger, WarnOnlyOnce
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
    """A backend model wraps the details to initialize and execute a 
    inference backend."""

    def __init__(self,
                 backend_files: Union[str, Sequence[str]],
                 backend: Optional[Backend] = None,
                 device_type: str = 'cpu',
                 device_id: int = 0,
                 force_cast: bool = False,
                 **kwargs):
        """The default methods to build a inference backend.
        Args:
            backend_files (Sequence[str]): Paths to all required backend files(
                e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            backend (Backend | None): The backend enum type.
            device_type (str): A string specifying device type. 
                Defaults to 'cpu'.
            device_id (int): A number specifying device id. Defaults to 0.
            force_cast (bool): Whether forcefully casting the input dtype to 
                the dtype specified by the backend.
        """
        self.logger = get_root_logger()
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
                    f"Automatically specify the `{backend}` backend for "
                    f"'{Path(backend_files).name}' backend-file.")
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
                                                 device_id, **kwargs)
        self.backend_mgr = backend_mgr
        self.force_cast = force_cast
        backend_file = backend_files if isinstance(backend_files, str) else \
                                         backend_files[0]
        self._name = Path(backend_file).name

    def destroy(self):
        if hasattr(self, 'backend') and hasattr(self.backend, 'destroy'):
            self.backend.destroy()
    
    def __call__(self, inputs: Union[Tensor, Sequence[Tensor]]):
        if isinstance(inputs, Tensor): 
            inputs = [inputs]
        if self.force_cast and self.backend.input_specs is not None:
            cast_inputs = []
            for input, input_spec in zip(inputs, self.backend.input_specs):
                if input.dtype is not None and input.dtype != input_spec.dtype:
                    WarnOnlyOnce.warn(
                        self.logger,
                        f"Backend model '{self._name}' requires "
                        f"{input_spec.dtype} for input "
                        f"`{input_spec.name}`. Cast {input.dtype} "
                        f"input to {input_spec.dtype}")
                    input = input.to(input_spec.dtype)
                cast_inputs.append(input)
            inputs = cast_inputs
        outputs = self.backend.infer(inputs)
        outputs = self.backend.output_to_sequence(outputs)
        return outputs

    def __repr__(self) -> str:
        repr_str = "****Env Info****\n"
        env_info = self.backend_mgr.check_env()
        repr_str += env_info + '\n'
        repr_str += repr(self.backend)
        return repr_str
