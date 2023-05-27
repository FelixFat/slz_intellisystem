import cv2
import numpy as np


def call(c_mask: np.ndarray, d_mask: np.ndarray) -> np.ndarray:
    res_mask = c_mask + d_mask
    res_mask = cv2.threshold(res_mask, 1, 1, cv2.THRESH_BINARY)[1].astype('uint8')

    return res_mask