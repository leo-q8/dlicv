from .backend_manager import AscendManager

__all__ = ['AscendManager']

if AscendManager.is_available():
    from .backend import AscendBackend
    __all__ += ['AscendBackend']
