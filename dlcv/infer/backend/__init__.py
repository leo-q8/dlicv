from enum import Enum

from .backend_model import BackendModel

class AdvancedEnum(Enum):
    """Define an enumeration class."""

    @classmethod
    def get(cls, value):
        """Get the key through a value."""
        for k in cls:
            if k.value == value:
                return k

        raise KeyError(f'Cannot get key by value "{value}" of {cls}')


class Backend(AdvancedEnum):
    """Define backend enumerations."""
    TENSORRT = 'tensorrt'
    ONNXRUNTIME = 'onnxruntime'
    NCNN = 'ncnn'
    OPENVINO = 'openvino'
    TORCHSCRIPT = 'torchscript'
    ASCEND = 'ascend'
    COREML = 'coreml'
    DEFAULT = 'default'

__all__ = ["BackendModel", "AdvancedEnum", "Backend"]