from typing import Sequence

from ais_bench.infer.interface import InferSession
import numpy as np

from dlcv.utils import Backend
from dlcv.utils.timer import TimeCounter
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
        output_names (Sequence[str] | None): Names of model outputs  in order.
            Defaults to `None` and the wrapper will load the output names from
            model.
    Note:
        If the engine is converted from onnx model. The input_names and
        output_names should be the same as onnx model.
    """

    def __init__(self, model_file: str, device_id: int = 0):
        self.sess = InferSession(device_id, model_file)
        input_specs, output_specs = [], []
        for i, input_desc in enumerate(self.sess.get_inputs()):
            input_specs.append(
                BackendIOSpec(input_desc.name, i, input_desc.shape, 
                       _from_acl_data_type[input_desc.datatype])
            )

        for i, output_desc in enumerate(self.sess.get_outputs()):
            output_specs.append(
                BackendIOSpec(output_desc.name, i, output_desc.shape, 
                       _from_acl_data_type[output_desc.datatype])
            )
        super().__init__([model_file], input_specs, output_specs)

    def _infer(self, inputs: Sequence[np.ndarray]) -> Backend_IOType:
        """Do inference.
        Args:
            inputs (Dict[str, np.ndarray]): The input name and tensor pairs.
        Return:
            Dict[str, np.ndarray]: The output name and tensor pairs.
        """
        outputs = self.__ascend_execute(inputs)
        return {i.name: output 
                for i, output in zip(self._output_specs, outputs)}
         
    @TimeCounter.count_time(Backend.ASCEND.value)
    def __ascend_execute(self, inputs: Sequence[np.ndarray]):
        """Run inference with ascend.
        Args:
            inputs (list[int]): A list of input array.
        """
        return self.sess.infer(inputs)
    