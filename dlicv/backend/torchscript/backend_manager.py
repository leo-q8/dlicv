import os.path as osp
from typing import Callable, Optional, Sequence

from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('torchscript')
class TorchScriptManager(BaseBackendManager):
    """This class is modified from `mmdeploy`
    https://github.com/open-mmlab/mmdeploy/blob/main/mmdeploy/backend/torchscript/backend_manager.py
    """

    @classmethod
    def build_backend(cls,
                      backend_files: Sequence[str], 
                      device_type: str = 'cpu',
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
            deploy_cfg (Optional[Any], optional): The deploy config. Defaults
                to None.
        """
        from .backend import TorchScriptBackend
        backend_file = osp.expanduser(backend_files[0])
        if not osp.isfile(backend_file):
            raise FileNotFoundError(f'`{backend_file}` not found.')
        return TorchScriptBackend(model_file=backend_file, 
                                  device_type=device_type,
                                  device_id=device_id,
                                  input_names=input_names,
                                  output_names=output_names)

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.
        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        return importlib.util.find_spec('torch') is not None and \
               importlib.util.find_spec('torchvision') is not None

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('torch').version
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
                torch_version = pkg_resources.get_distribution(
                    'torch').version
                torchvision_version = pkg_resources.get_distribution(
                        'torchvision').version
                torch_info = f'torch:\t{torch_version}'
                log_callback(torch_info)
                torchvision_info = f'torchvision:\t{torchvision_version}'
                log_callback(torchvision_info)
                info = f'{torch_info}\n{torchvision_info}'
        except Exception:
            info = f'{cls.backend_name}:\tCheckFailed'
            log_callback(info)
        return info