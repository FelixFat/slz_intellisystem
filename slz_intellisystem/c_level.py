import cv2
import numpy as np


M_SHAPE = (384, 384)
C_SHAPE = (640, 480)


def get_mask(pr_mask):
    mask = pr_mask.squeeze()[..., 0] + pr_mask.squeeze()[..., 1]
    mask = cv2.resize(mask, C_SHAPE)
    mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
    return mask


def call(model, image):
    im = cv2.resize(image, M_SHAPE)
    im = np.expand_dims(im, axis=0)
    pr_mask = model.predict(im)

    mask = get_mask(pr_mask)

    return mask
