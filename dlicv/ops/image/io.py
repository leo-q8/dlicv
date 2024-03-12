import os.path as osp
from typing import Union, Optional

import cv2
import numpy as np
from cv2 import (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION,
                 IMREAD_UNCHANGED)
from pathlib import Path

from dlicv.utils import mkdir_or_exist

try:
    from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
except ImportError:
    TJCS_RGB = TJPF_GRAY = TJPF_BGR = TurboJPEG = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None

try:
    import tifffile
except ImportError:
    tifffile = None

imread_backends = ['cv2', 'turbojpeg', 'pillow', 'tifffile']

cv2_imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED,
    'color_ignore_orientation': IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    'grayscale_ignore_orientation':
    IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE
}


def _check_imread_backend(backend: str) -> None:
    """Select a backend for image decoding.

    Args:
        backend (str): The image decoding backend type. Options are `cv2`,
        `pillow`, `turbojpeg` (see https://github.com/lilohuang/PyTurboJPEG)
        and `tifffile`. `turbojpeg` is faster but it only supports `.jpeg`
        file format.
    """
    assert backend in imread_backends, ("Supported backends are 'cv2', "
                                        "turbojpeg', 'pillow', 'tifffile'")
    if backend == 'turbojpeg':
        if TurboJPEG is None:
            raise ImportError('`PyTurboJPEG` is not installed')
    elif backend == 'pillow':
        if Image is None:
            raise ImportError('`Pillow` is not installed')
    elif backend == 'tifffile':
        if tifffile is None:
            raise ImportError('`tifffile` is not installed')


def _jpegflag(flag: str = 'color', channel_order: str = 'bgr'):
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'color':
        if channel_order == 'bgr':
            return TJPF_BGR
        elif channel_order == 'rgb':
            return TJCS_RGB
    elif flag == 'grayscale':
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')


def _pillow2array(img,
                  flag: str = 'color',
                  channel_order: str = 'bgr') -> np.ndarray:
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'unchanged':
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # Handle exif orientation tag
        if flag in ['color', 'grayscale']:
            img = ImageOps.exif_transpose(img)
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag in ['color', 'color_ignore_orientation']:
            array = np.array(img)
            if channel_order != 'rgb':
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag in ['grayscale', 'grayscale_ignore_orientation']:
            img = img.convert('L')
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale", "unchanged", '
                f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
                f' but got {flag}')
    return array


def imread(img_path: Union[str, Path],
           flag: str = 'color',
           channel_order: str = 'bgr',
           backend: str = 'cv2') -> np.ndarray:
    """Read an image.

    Args:
        img_path (str or Path): Either str or pathlib.Path. 
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale`, `unchanged`,
            `color_ignore_orientation` and `grayscale_ignore_orientation`.
            By default, `cv2` and `pillow` backend would rotate the image
            according to its EXIF info unless called with `unchanged` or
            `*_ignore_orientation` flags. `turbojpeg` and `tifffile` backend
            always ignore image's EXIF info regardless of the flag.
            The `turbojpeg` backend only supports `color` and `grayscale`.
        channel_order (str): Order of channel, candidates are `bgr` and `rgb`.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`.
            If backend is None, the global imread_backend specified by
            ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        ndarray: Loaded image array.
    """
    
    if isinstance(img_path, Path):
        img_path = str(img_path)
    
    _check_imread_backend(backend)

    if backend == 'turbojpeg':
        jpeg = TurboJPEG()
        with open(img_path, 'rb') as f:
            img = jpeg.decode(f.read(), _jpegflag(flag, channel_order))
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        return img
    elif backend == 'pillow':
        img = Image.open(img_path)
        img = _pillow2array(img, flag, channel_order)
        return img
    elif backend == 'tifffile':
        img = tifffile.imread(img_path)
        return img
    else:
        flag = cv2_imread_flags[flag] if isinstance(flag, str) else flag
        img = cv2.imread(img_path, flag)
        if flag == IMREAD_COLOR and channel_order == 'rgb':
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img


def imwrite(file_path: Union[str, Path],
            img: np.ndarray,
            params: Optional[list] = None) -> bool:
    """Write image to file.

    Args:
        file_path (str): Image file path.
        img (ndarray): Image array to be written.
        params (None or list): Same as opencv :func:`imwrite` interface.

    Returns:
        bool: Successful or not.

    """
    file_path = str(file_path)
    img_ext = osp.splitext(file_path)[-1]
    # Encode image according to image suffix.
    # For example, if image path is '/path/your/img.jpg', the encode
    # format is '.jpg'.
    flag, img_buff = cv2.imencode(img_ext, img, params)
    
    mkdir_or_exist(osp.dirname(file_path))
    with open(file_path, 'wb') as f:
        f.write(img_buff.tobytes())
    return flag