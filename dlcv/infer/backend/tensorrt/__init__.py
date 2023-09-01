# flake8: noqa
from .backend_manager import TensorRTManager

__all__ = ['TensorRTManager']

if TensorRTManager.is_available():
    try:
        from .backend import TRTBackend
        __all__ += ['TRTBackend']
    except Exception:
        pass
