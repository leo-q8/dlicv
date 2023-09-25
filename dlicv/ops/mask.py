import cv2
import numpy as np


retrieval_modes = {
    'external': cv2.RETR_EXTERNAL,
    'list': cv2.RETR_LIST,
    'ccomp': cv2.RETR_CCOMP,
    'tree': cv2.RETR_TREE,
}

chain_approx_method = {
    'none': cv2.CHAIN_APPROX_NONE,
    'simple': cv2.CHAIN_APPROX_SIMPLE,
    'l1': cv2.CHAIN_APPROX_TC89_L1,
    'kcos': cv2.CHAIN_APPROX_TC89_KCOS
}

def bitmap_to_polygon(bitmap: np.ndarray, 
                      mode: str = 'ccomp',
                      method: str =  'none'):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.
        mode (str): cv2 

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    assert mode in retrieval_modes.keys() and \
        method in chain_approx_method.keys()
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, 
                            retrieval_modes[mode],
                            chain_approx_method[method])
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole