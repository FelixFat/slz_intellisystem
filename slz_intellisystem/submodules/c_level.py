import cv2
import numpy as np

from .params import IMG_SHAPE, MOD_SHAPE


def get_mask(pr_mask):
    mask = pr_mask.squeeze()[..., 0] + pr_mask.squeeze()[..., 1]
    mask = cv2.resize(mask, (IMG_SHAPE[1], IMG_SHAPE[0]))
    mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
    return mask


def call(model, image):
    im = cv2.resize(image, MOD_SHAPE)
    im = np.expand_dims(im, axis=0)
    pr_mask = model.predict(im)

    mask = get_mask(pr_mask)

    return mask
