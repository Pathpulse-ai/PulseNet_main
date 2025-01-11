import torch
import torchvision.ops as ops

def box_area(boxes):
    """
    Computes area of boxes in [x1, y1, x2, y2] format.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def iou(boxes1, boxes2):
    """
    Computes Intersection-over-Union (IoU) between two sets of boxes.
    """
    return ops.box_iou(boxes1, boxes2)

def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-maximum suppression wrapper around torchvision ops.
    """
    keep = ops.nms(boxes, scores, iou_threshold)
    return keep