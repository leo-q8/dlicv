import os.path as osp
from typing import Optional, Sequence, Callable

from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('onnxruntime')
class ONNXRuntimeManager(BaseBackendManager):
    """This class is modified from `mmdepoly`
    https://github.com/open-mmlab/mmdeploy/blob/main/mmdeploy/backend/onnxruntime/backend_manager.py
    """

    @classmethod
    def build_backend(cls,
                      backend_files: Sequence[str], 
                      device_type: str = 'cpu',
                      device_id: int = 0,
                      input_names: Optional[Sequence[str]] = None,
                      output_names: Optional[Sequence[str]] = None,
                      **kwargs):
        """Build the backend for the backend model.
        Args:
            backend_files (Sequence[str]): Backend files.
            device_type (str): A string specifying device type. 
                Defaults to 'cpu'.
            device_id (int): A number specifying device id. Defaults to 0.
            input_names (Optional[Sequence[str]], optional): input names.
                Defaults to None.
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
        """
        from .backend import ORTBackend
        return ORTBackend(model_file=backend_files[0],
                          device_type=device_type,
                          device_id=device_id)

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.
        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        return importlib.util.find_spec('onnxruntime') is not None

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                ort_version = pkg_resources.get_distribution(
                    'onnxruntime').version
            except Exception:
                ort_version = 'None'
            try:
                ort_gpu_version = pkg_resources.get_distribution(
                    'onnxruntime-gpu').version
            except Exception:
                ort_gpu_version = 'None'

            if ort_gpu_version != 'None':
                return ort_gpu_version
            else:
                return ort_version
    
    @classmethod
    def check_env(cls, log_callback: Callable = lambda _: _) -> str:
        """Check current environment.

        Returns:
            str: Info about the environment.
        """
        import pkg_resources

        try:
            if cls.is_available():
                try:
                    ort_version = pkg_resources.get_distribution(
                        'onnxruntime').version
                except Exception:
                    ort_version = 'None'
                try:
                    ort_gpu_version = pkg_resources.get_distribution(
                        'onnxruntime-gpu').version
                except Exception:
                    ort_gpu_version = 'None'

                ort_info = f'ONNXRuntime:\t{ort_version}'
                log_callback(ort_info)
                ort_gpu_info = f'ONNXRuntime-gpu:\t{ort_gpu_version}'
                log_callback(ort_gpu_info)

                info = f'{ort_info}\n{ort_gpu_info}'
            else:
                info = 'ONNXRuntime:\tNone'
                log_callback(info)
        except Exception:
            info = f'{cls.backend_name}:\tCheckFailed'
            log_callback(info)
        return info