import cv2
import numpy as np

from .params import IMG_SHAPE


def calc_dist(in_pos: np.ndarray):
    return np.linalg.norm((IMG_SHAPE[0] // 2, IMG_SHAPE[1] // 2) - in_pos)


def find_nearest(points: np.ndarray):
    return points[np.argmin([calc_dist(cc) for cc in points])]


def call(mask):
    res_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value=0)
    res_mask = cv2.distanceTransform(res_mask, cv2.DIST_L2, 3)
    res_mask = res_mask[1:res_mask.shape[0]-1, 1:res_mask.shape[1]-1]

    radius = np.max(res_mask)
    center = find_nearest(
        np.column_stack(np.where(res_mask == radius))
    )
    x, y = center

    return res_mask, x, y, radius