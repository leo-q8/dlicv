from typing import Optional, Sequence

import torch
import torchvision # necessary for load `torchvison.ops.nsm`

from dlcv.utils import Backend
from dlcv.utils.timer import TimeCounter
from ..base import BaseBackend, BackendIOSpec, Backend_IOType


class TorchScriptBackend(BaseBackend):
    """Torchscript engine wrapper for inference.
    Args:
        model (torch.jit.RecursiveScriptModule): torchscript engine to wrap.
        input_names (Sequence[str] | None): Names of model inputs  in order.
            Defaults to `None` and the wrapper will accept list or Tensor.
        output_names (Sequence[str] | None): Names of model outputs  in order.
            Defaults to `None` and the wrapper will return list or Tensor.
    Note:
        If the engine is converted from onnx model. The input_names and
        output_names should be the same as onnx model.
    Examples:
        >>> from mmdeploy.backend.torchscript import TorchscriptWrapper
        >>> engine_file = 'resnet.engine'
        >>> model = TorchscriptWrapper(engine_file, input_names=['input'], \
        >>>    output_names=['output'])
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self,
                 model_file: str, 
                 device_type: str = 'cpu',
                 device_id: Optional[int] = None,
                 input_names: Optional[Sequence[str]] = None,
                 output_names: Optional[Sequence[str]] = None):
        assert device_type in ('cuda', 'cpu'), f'Only support `cuda` or `cpu`'\
            f' device_type, but got `{device_type}`'
        if device_type == 'cuda' and device_id is not None:
            self.device = f'{device_type}:{device_id}'
        else:
            self.device = device_type
        ts_model = torch.jit.load(model_file, map_location='cpu')
        assert isinstance(ts_model, torch.jit.RecursiveScriptModule
                          ), 'failed to load torchscript model.'
        self.ts_model = ts_model.to(self.device)
        if input_names is not None:
            input_specs = []
            for i, name in enumerate(input_names):
                input_specs.append(BackendIOSpec(name, i, None, None))
        else:
            input_specs = None

        if output_names is not None:
            output_specs = []
            for i, name in enumerate(output_names):
                output_specs.append(BackendIOSpec(name, i, None, None))
        else:
            output_specs = None
        super().__init__([model_file], input_specs, output_specs)

    def _infer(self, inputs: Sequence[torch.Tensor]) -> Backend_IOType:
        """Do inference.
        Args:
            inputs (Dict[str, np.ndarray]): The input name and tensor pairs.
        Return:
            Dict[str, np.ndarray]: The output name and tensor pairs.
        """
        torch_inputs = [x.to(self.device) for x in inputs]
        outputs = self.__torchscript_execute(torch_inputs)
        if self._output_specs is not None:
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            assert len(outputs) == len(self._output_specs)
            outputs = {i.name: output
                       for i, output in zip(self._output_specs, outputs)}
        return outputs

    @torch.no_grad()
    @TimeCounter.count_time(Backend.TORCHSCRIPT.value)
    def __torchscript_execute(
            self, inputs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """Run inference with TorchScript.
        Args:
            inputs (Sequence[torch.Tensor]): A list of integer binding the
            input/output.
        Returns:
            torch.Tensor | Sequence[torch.Tensor]: The inference outputs from
            TorchScript.
        """
        return self.ts_model(*inputs)