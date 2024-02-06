import os.path as osp
from typing import Callable, Optional, Sequence

from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('ascend')
class AscendManager(BaseBackendManager):
    """This class is modified from `mmdepoly`
    https://github.com/open-mmlab/mmdeploy/blob/main/mmdeploy/backend/ascend/backend_manager.py
    """

    @classmethod
    def build_backend(cls,
                      backend_files: Sequence[str], 
                      device: str = 'cpu',
                      device_id: int = 0,
                      input_names: Optional[Sequence[str]] = None,
                      output_names: Optional[Sequence[str]] = None,
                      **kwargs):
        """Build the wrapper for the backend model.
        Args:
            backend_files (Sequence[str]): Backend files.
            device (str, optional): The device info. Defaults to 'cpu'.
            input_names (Optional[Sequence[str]], optional): input names.
                Defaults to None.
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
        """
        from .backend import AscendBackend
        backend_file = osp.expanduser(backend_files[0])
        if not osp.isfile(backend_file):
            raise FileNotFoundError(f'`{backend_file}` not found.')
        return AscendBackend(model_file=backend_file, device_id=device_id)

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.
        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        return importlib.util.find_spec('aclruntime') is not None and \
               importlib.util.find_spec('ais_bench') is not None

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('ais_bench').version
            except Exception:
                return 'None'
    
    @classmethod
    def check_env(cls, log_callback: Callable = lambda _: _) -> str:
        """Check current environment.

        Returns:
            str: Info about the environment.
        """
        import pkg_resources

        try:
            if cls.is_available():
                aclruntime_version = pkg_resources.get_distribution(
                    'aclruntime').version
                ais_bench_version = pkg_resources.get_distribution(
                        'ais_bench').version
                aclruntime_info = f'aclruntime:\t{aclruntime_version}'
                log_callback(aclruntime_info)
                ais_bench_info = f'ais_bench:\t{ais_bench_version}'
                log_callback(ais_bench_info)
                info = f'{aclruntime_info}\n{ais_bench_info}'
        except Exception:
            info = f'{cls.backend_name}:\tCheckFailed'
            log_callback(info)
        return info
