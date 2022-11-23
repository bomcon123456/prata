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

def iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
     
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou

def is_in_box(pnt, box):
    if pnt[0] >= box[0] and pnt[0] <= box[0]+box[2] and pnt[1] >= box[1] and pnt[1] <= box[1]+box[3]:
        return True
    else:
        return False