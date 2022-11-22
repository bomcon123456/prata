from cython_bbox import bbox_overlaps as bbox_ious
import numpy as np

def ious(lboxes, rboxes):
    """
    Compute cost based on IoU
    :type lboxes: list[ltrb] | np.ndarray
    :type rboxes: list[ltrb] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(lboxes), len(rboxes)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(lboxes, dtype=np.float),
        np.ascontiguousarray(rboxes, dtype=np.float),
    )

    return ious