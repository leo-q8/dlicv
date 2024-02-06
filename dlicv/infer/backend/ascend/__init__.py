from .backend_manager import AscendManager

__all__ = ['AscendManager']

if AscendManager.is_available():
    try:
        from .backend import AscendBackend
        __all__ += ['AscendBackend']
    except Exception:
        pass
