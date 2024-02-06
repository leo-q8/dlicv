from typing import Sequence, Dict

from ais_bench.infer.interface import InferSession
import numpy as np
import torch
from torch import Tensor

from dlicv.utils import Backend
from dlicv.utils.timer import TimeCounter
from ..base import BaseBackend, BackendIOSpec, Backend_IOType

_from_acl_data_type = {
    0: np.float32, 
    1: np.float16, 
    2: np.int8,
    3: np.int32, 
    6: np.int16,
    9: np.int64
}

class AscendBackend(BaseBackend):
    """Ascend backbend for inference.
    Args:
        model_file (str): Ascend off-line inference model file.
        device_id (int): A number specifying ascend device id. 
            Defaults to 0.
    """

    def __init__(self, model_file: str, device_id: int = 0):
        self.sess = InferSession(device_id, model_file)
        input_specs, output_specs = [], []
        for i, input_desc in enumerate(self.sess.get_inputs()):
            input_specs.append(
                BackendIOSpec(i, input_desc.name, input_desc.shape, 
                              _from_acl_data_type[input_desc.datatype]))

        for i, output_desc in enumerate(self.sess.get_outputs()):
            output_specs.append(
                BackendIOSpec(i, output_desc.name, output_desc.shape, 
                              _from_acl_data_type[output_desc.datatype]))
        super().__init__([model_file], input_specs, output_specs)

    def _infer(self, inputs: Sequence[Tensor]) -> Dict[str, Tensor]:
        """Do inference.
        Args:
            inputs (Sequence[Tensor]): The input tensor sequence.
        Return:
            Dict[str, Tensor]: The output name and tensor pairs.
        """
        inputs = [input.cpu().numpy() for input in inputs]
        outputs = self.__ascend_execute(inputs)
        return {i.name: torch.from_numpy(output)
                for i, output in zip(self._output_specs, outputs)}
         
    @TimeCounter.count_time(Backend.ASCEND.value)
    def __ascend_execute(self, inputs: Sequence[Tensor]):
        """Run inference with ascend.
        Args:
            inputs (Sequence[Tensor]): A sequence of input Tensor.
        """
        return self.sess.infer(inputs)
    