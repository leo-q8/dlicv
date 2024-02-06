from .backend_manager import ONNXRuntimeManager

__all__ = ['ONNXRuntimeManager']

if ONNXRuntimeManager.is_available():
    try:
        from .backend import ORTBackend
        __all__ += ['ORTBackend']
    except Exception:
        pass