from typing import Dict, Sequence

import numpy as np
import onnxruntime as ort
import torch

from dlicv.utils import Backend
from dlicv.utils.timer import TimeCounter
from ..base import BaseBackend, BackendIOSpec


def get_torch_dtype(dtype: str):
    if 'int8' in dtype:
        return torch.int8
    elif 'int32' in dtype:
        return torch.int32
    elif 'int64' in dtype:
        return torch.int64
    elif 'float16' in dtype:
        return torch.float16
    elif 'float32' in  dtype:
        return torch.float32
    elif '(float)' in dtype:
        return torch.float32
    else:
        raise TypeError(f'{dtype} is not supported by torch')


class ORTBackend(BaseBackend):
    """ONNXRuntime backend for inference. This class is modified from 
    `mmdeploy` https://github.com/open-mmlab/mmdeploy/blob/main/mmdeploy/backend/onnxruntime/wrapper.py

     Args:
         model_file (str): Input onnx model file.
         device_type (str): A string specifying device type. 
                Defaults to 'cpu'.
         device_id (int): A number specifying device id. Defaults to 0.

     Examples:
         >>> from dlicv.backend.onnxruntime import ORTBackend
         >>> import torch
         >>>
         >>> model_file = 'model.onnx'
         >>> model = ORTBackend(model_file)
         >>> inputs = torch.randn(1, 3, 224, 224, device='cpu')
         >>> outputs = model(inputs)
         >>> print(outputs)
    """
    
    def __init__(self,
                 model_file: str, 
                 device_type: str = 'cpu',
                 device_id: int = 0):
        assert device_type in ('cuda', 'cpu'), f'Only support `cuda` or '\
            f'`cpu` device_type, but got `{device_type}`'
        providers = ['CPUExecutionProvider'] \
            if device_type == 'cpu' else  [('CUDAExecutionProvider', 
                                           {'device_id': device_id})]
        sess = ort.InferenceSession(
            model_file, providers=providers)

        input_specs, output_specs = [], []
        input_types, output_types = [], []
        for index, input in enumerate(sess.get_inputs()):
            torch_dtype = get_torch_dtype(input.type)
            input_specs.append(
                BackendIOSpec(index, input.name, 
                              tuple(input.shape), torch_dtype))
            input_types.append(
                torch.tensor(1, dtype=torch_dtype).numpy().dtype)
        for index, output in enumerate(sess.get_outputs()):
            output_specs.append(
                BackendIOSpec(index, output.name, tuple(output.shape), 
                              get_torch_dtype(output.type)))
            output_types.append(
                torch.tensor(1, dtype=torch_dtype).numpy().dtype)
        self.sess = sess
        self.io_binding = sess.io_binding()
        self.input_types, self.output_types = input_types, output_types 
        self.device_type = device_type
        self.device_id = device_id
        self.torch_device = torch.device(device_type, device_id)
        super().__init__([model_file], input_specs, output_specs)

    def _infer(self, inputs: Sequence[torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
        """Do inference.
        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.
        Return:
            Dict[str, torch.Tensor]: The output name and tensor pairs.
        """
        for input_tensor, input_spec, input_dtype in zip(
                inputs, self._input_specs, self.input_types):
            assert input_tensor.dtype == input_spec.dtype, \
                f'Require {input_spec.dtype} for `{input_spec.name}`, '\
                f'but get {input_tensor.dtype}.'
            input_tensor = input_tensor.to(self.torch_device).contiguous()
            self.io_binding.bind_input(
                name=input_spec.name,
                device_type=self.device_type,
                device_id=self.device_id,
                element_type=input_dtype,
                shape=input_tensor.shape,
                buffer_ptr=input_tensor.data_ptr())
        
        for output_spec in self._output_specs:
            self.io_binding.bind_output(output_spec.name)

        # run session to get outputs
        if self.device_type == 'cuda':
            torch.cuda.synchronize()
        self.__ort_execute(self.io_binding)
        output_list = self.io_binding.copy_outputs_to_cpu()
        outputs = {}
        for output_spec, numpy_tensor in zip(self._output_specs, output_list):
            if numpy_tensor.dtype == np.float16:
                numpy_tensor = numpy_tensor.astype(np.float32)
            outputs[output_spec.name] = torch.from_numpy(numpy_tensor)

        return outputs

    @TimeCounter.count_time(Backend.ONNXRUNTIME.value)
    def __ort_execute(self, io_binding: ort.IOBinding):
        """Run inference with ONNXRuntime session.

        Args:
            io_binding (ort.IOBinding): To bind input/output to a specified
                device, e.g. GPU.
        """
        self.sess.run_with_iobinding(io_binding)