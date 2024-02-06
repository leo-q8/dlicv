import glob
import logging
import os

from dlicv.utils.logging import get_logger


class WarnOnlyOnce:
    """Warning only once when getting the same warning message. 
    """
    warnings = set()

    @classmethod
    def warn(cls, logger: logging.Logger, msg: str):
        h = hash(msg)
        if h not in cls.warnings:
            logger.warning(msg)
            cls.warnings.add(h)


def get_root_logger(log_file=None, log_level=logging.INFO) -> logging.Logger:
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
    Returns:
        logging.Logger: The obtained logger
    """
    logger = get_logger(
        name='dlicv', log_file=log_file, log_level=log_level)

    return logger


def get_file_path(prefix, candidates) -> str:
    """Search for file in candidates.

    Args:
        prefix (str): Prefix of the paths.
        candidates (str): Candidate paths
    Returns:
        str: file path or '' if not found
    """
    for candidate in candidates:
        wildcard = os.path.abspath(os.path.join(prefix, candidate))
        paths = glob.glob(wildcard)
        if paths:
            lib_path = paths[0]
            return lib_path
    return ''