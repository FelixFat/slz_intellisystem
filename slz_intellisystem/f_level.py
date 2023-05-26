import cv2
import numpy as np


IMG_SHAPE = (480, 640)
IMG_CENTER = (240, 320)


def calc_dist(in_pos: np.ndarray):
    return np.linalg.norm(IMG_CENTER - in_pos)


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

    return res_mask, center, radius