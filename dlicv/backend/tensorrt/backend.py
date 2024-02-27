# This file is modified from `mmdeploy`
# https://github.com/open-mmlab/mmdeploy/blob/main/mmdeploy/backend/tensorrt/wrapper.py
from typing import Sequence, Dict

import tensorrt as trt
import torch
from torch import Tensor

from dlicv.utils import Backend
from dlicv.utils.timer import TimeCounter
from ..base.base_backend import BackendIOSpec, BaseBackend
from .torch_allocator import TorchAllocator


def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
    """Convert pytorch dtype to TensorRT dtype.

    Args:
        dtype (str.DataType): The data type in tensorrt.

    Returns:
        torch.dtype: The corresponding data type in torch.
    """

    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f'{dtype} is not supported by torch')

        
def torch_device_from_trt(device: trt.TensorLocation):
    """Convert pytorch device to TensorRT device.

    Args:
        device (trt.TensorLocation): The device in tensorrt.
    Returns:
        torch.device: The corresponding device in torch.
    """
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError(f'{device} is not supported by torch')


class TRTBackend(BaseBackend):
    """TensorRT backbend for inference.
    Args:
        engine_file (str): TensorRT engine file.
        device_id (int): A number specifying the cuda device id.

    Examples:
        >>> from dlicv.backend.tensorrt import TRTBackend
        >>> engine_file = 'resnet.engine'
        >>> model = TRTBackend(engine_file)
        >>> inputs = torch.randn(1, 3, 224, 224).cuda()
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self, engine_file: str, device_id: int = 0):
        self.torch_device = torch.device('cuda', device_id)
        self.allocator = TorchAllocator(device_id)
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(engine_file, mode='rb') as f:
                engine_bytes = f.read()
            trt.init_libnvinfer_plugins(logger, namespace='')
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()

        if hasattr(self.context, 'temporary_allocator'):
            self.context.temporary_allocator = self.allocator

        self.__load_io_names()
        
        input_specs, output_specs = [], []
        profile_id = 0
        for input_name in self._input_names:
            index = self.engine.get_binding_index(input_name)
            dtype = torch_dtype_from_trt(
                self.engine.get_binding_dtype(input_name)) 
            shape = tuple(self.context.get_binding_shape(index))
            if -1 in shape:
                profile_shape = self.engine.get_profile_shape(profile_id,
                                                              input_name)
                shape = {'min': profile_shape[0], 
                         'opt': profile_shape[1],
                         'max': profile_shape[2]}
            input_specs.append(
                BackendIOSpec(index, input_name, shape, dtype))

        for output_name in self._output_names:
            index = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(
                self.engine.get_binding_dtype(output_name)) 
            shape = tuple(self.context.get_binding_shape(index))
            output_specs.append(
                BackendIOSpec(index, output_name, shape, dtype))
        super().__init__([engine_file], input_specs, output_specs)
    
    def __load_io_names(self):
        """Load input/output names from engine."""
        input_names, output_names = [], []
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                input_names.append(binding)
            else:
                output_names.append(binding)
        self._input_names, self._output_names = input_names, output_names

    def _infer(self, inputs: Sequence[Tensor]) -> Dict[str, Tensor]:
        """Do inference.
        Args:
            inputs (Sequence[torch.Tensor]): The input tensor sequence.

        Return:
            Dict[str, torch.Tensor]: The output name and tensor pairs.
        """
        # Make self the active context, pushing it on top of the context stack.
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile_id = 0
        for input_spec, input_tensor in zip(self._input_specs, inputs):
            index, name, _, dtype = input_spec
            profile = self.engine.get_profile_shape(profile_id, name)
            # check if input shape is valid
            assert input_tensor.ndim == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape, 
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f"{profile[0]} and {profile[2]}" \
                    + f' but get {tuple(input_tensor.shape)}.'

            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            input_tensor = input_tensor.contiguous()
            assert input_tensor.dtype == dtype, \
                f'Require {dtype} for `{name}`, but get {input_tensor.dtype}.'
            input_tensor = input_tensor.to(self.torch_device)
            self.context.set_binding_shape(index, tuple(input_tensor.shape))
            bindings[index] = input_tensor.contiguous().data_ptr()
            
        # Collect output array
        outputs = {}
        for output_spec in self._output_specs:
            index, name, _, dtype = output_spec
            device = torch_device_from_trt(self.engine.get_location(index))
            shape = tuple(self.context.get_binding_shape(index))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[name] = output
            bindings[index] = output.data_ptr()

        # Run inference.
        self.__trt_execute(bindings=bindings)

        return outputs

    @TimeCounter.count_time(Backend.TENSORRT.value)
    def __trt_execute(self, bindings: Sequence[int]):
        """Run inference with TensorRT.
        Args:
            bindings (list[int]): A list of integer binding the input/output.
        """
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
