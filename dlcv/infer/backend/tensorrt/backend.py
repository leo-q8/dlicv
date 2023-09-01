from typing import Sequence

import numpy as np
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from backend import Backend
from utils.timer import TimeCounter
from ..base.base_backend import BackendIOSpec, Backend_IOType, BaseBackend


class TRTBackend(BaseBackend):
    """TensorRT backbend for inference.
    Args:
        engine (tensorrt.ICudaEngine): TensorRT engine to wrap.
        output_names (Sequence[str] | None): Names of model outputs  in order.
            Defaults to `None` and the wrapper will load the output names from
            model.
    Note:
        If the engine is converted from onnx model. The input_names and
        output_names should be the same as onnx model.
    Examples:
        >>> from mmdeploy.backend.tensorrt import TRTWrapper
        >>> engine_file = 'resnet.engine'
        >>> model = TRTWrapper(engine_file)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self, engine_file: str):
        self._dynamic=False
        self._input_allocs, self._output_allocs = [], [] 
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(engine_file, mode='rb') as f:
                engine_bytes = f.read()
            trt.init_libnvinfer_plugins(logger, namespace='')
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        self.__load_io_names()
        
        input_specs, output_specs = [], []
        input_allocs, output_allocs = {}, {}
        profile_id = 0
        for input_name in self._input_names:
            index = self.engine.get_binding_index(input_name)
            dtype = trt.nptype(self.engine.get_binding_dtype(input_name)) 
            dtype = np.dtype(dtype)
            shape = tuple(self.engine.get_binding_shape(input_name))
            if -1 in shape:
                self._dynamic = True
                profile_shape = self.engine.get_profile_shape(profile_id,
                                                              input_name)
                shape = {'min': profile_shape[0], 
                         'opt': profile_shape[1],
                         'max': profile_shape[2]}
                # Set input binding shape for geting output max shape
                self.context.set_binding_shape(index, shape['max'])
                alloc = cuda.mem_alloc(
                    trt.volume(shape['max']) * dtype.itemsize
                ) 
            else:
                alloc = cuda.mem_alloc(trt.volume(shape)*dtype.itemsize) 
            input_specs.append(
                BackendIOSpec(input_name, index, shape, dtype)
            )
            input_allocs[index] = alloc

        for output_name in self._output_names:
            index = self.engine.get_binding_index(output_name)
            dtype = trt.nptype(self.engine.get_binding_dtype(output_name)) 
            dtype = np.dtype(dtype)
            shape = tuple(self.engine.get_binding_shape(output_name))
            if self._dynamic:
                binding_shape = self.context.get_binding_shape(index)
                alloc = cuda.mem_alloc(
                    trt.volume(binding_shape) * dtype.itemsize
                ) 
            else:
                alloc = cuda.mem_alloc(trt.volume(shape) * dtype.itemsize) 
            output_specs.append(
                BackendIOSpec(output_name, index, shape, dtype)
            )
            output_allocs[index] = alloc
        super().__init__([engine_file], input_specs, output_specs)
        self._input_allocs = [input_allocs[k] for k in sorted(input_allocs)]
        self._output_allocs = [output_allocs[k] for k in sorted(output_allocs)]
    
    def __load_io_names(self):
        """Load input/output names from engine."""
        input_names, output_names = [], []
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                input_names.append(binding)
            else:
                output_names.append(binding)
        self._input_names, self._output_names = input_names, output_names

    def _infer(self, inputs: Sequence[np.ndarray]) -> Backend_IOType:
        """Do inference.
        Args:
            inputs (Dict[str, np.ndarray]): The input name and tensor pairs.
        Return:
            Dict[str, np.ndarray]: The output name and tensor pairs.
        """
        # Make self the active context, pushing it on top of the context stack.
        for input_spec, alloc in zip(self._input_specs, self._input_allocs):
            name, index, shape, dtype = input_spec
            input_array = inputs[index]
            assert input_array.dtype == dtype, \
                f'Require {dtype} for {name}, but get {input_array.dtype}.'
            if self._dynamic and isinstance(shape, dict): 
                # check if input shape is valid
                assert len(input_array.shape) == len(shape['min']), \
                    'Input dim is different from engine profile.'
                for s_min, s_input, s_max in zip(shape['min'], 
                                                 input_array.shape, 
                                                 shape['max']):
                    assert s_min <= s_input <= s_max, \
                        'Input shape should be between ' \
                        + f"{shape['min']} and {shape['max']}" \
                        + f' but get {tuple(input_array.shape)}.'
                self.context.set_binding_shape(index, input_array.shape)
            else:
                assert shape == tuple(input_array.shape), \
                    f'Input shape shoule equal to {shape} ' \
                    + f'but get {input_array.shape}.'
            # Copy input image to host buffer
            cuda.memcpy_htod(alloc, input_array)

        bindings = [int(alloc) for alloc in self._input_allocs + 
                                            self._output_allocs]
        # Run inference.
        self.__trt_execute(bindings=bindings)
        # Collect output array
        outputs = {}
        for output_spec, alloc in zip(self._output_specs, self._output_allocs):
            name, index, _, dtype = output_spec
            binding_shape = self.context.get_binding_shape(index)
            output_array = np.empty(binding_shape, dtype=dtype)
            cuda.memcpy_dtoh(output_array, alloc)
            outputs[name] = output_array

        return outputs

    @TimeCounter.count_time(Backend.TENSORRT.value)
    def __trt_execute(self, bindings: Sequence[int]):
        """Run inference with TensorRT.
        Args:
            bindings (list[int]): A list of integer binding the input/output.
        """
        self.context.execute_v2(bindings)
    
    def destroy(self):
        for alloc in self._input_allocs + self._output_allocs:
            alloc.free()
        
    def __del__(self):
        self.destroy()
