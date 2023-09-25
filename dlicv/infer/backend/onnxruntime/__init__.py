from .backend_manager import ONNXRuntimeManager

__all__ = ['ONNXRuntimeManager']

if ONNXRuntimeManager.is_available():
    from .backend import ORTBackend
    __all__ += ['ORTBackend']