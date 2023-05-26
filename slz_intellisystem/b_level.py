import numpy as np

from params import *
import c_level
import d_level
import e_level
import f_level


def rs_dmap2data(dmap: np.ndarray) -> np.ndarray:
    return dmap * MAX_DIST / 255


def px2m(r:float, h: float):
    delta = np.mean([
        np.deg2rad(C_FOV[0] / 2) * h / IMG_SHAPE[0],
        np.deg2rad(C_FOV[1] / 2) * h / IMG_SHAPE[1]
    ])
    return r * delta


def cals_area(in_radius: float, in_height: float):
    R = px2m(in_radius, in_height)
    return np.pi * (R ** 2)


def get_input(in_color, in_depth, in_hag_dist):
    depth = rs_dmap2data(in_depth)
    return in_color, depth, in_hag_dist


def disp(model, color, depth, hag_dist):

    valid = True
    etype = 0
    c_mask = np.zeros(IMG_SHAPE)
    d_mask = np.zeros(IMG_SHAPE)

    if HIGH_DIST < hag_dist and hag_dist <= MAX_DIST:
        # COLOR-only
        c_mask = c_level.call(model, color)
        d_mask = np.ones(IMG_SHAPE)
        etype = 1

    elif LOW_DIST < hag_dist and hag_dist <= HIGH_DIST:
        # COLOR & DEPTH
        c_mask = c_level.call(model, color)
        d_mask = d_level.call(depth)
        etype = 3

    elif MIN_DIST <= hag_dist and hag_dist <= LOW_DIST:
        # DEPTH-only
        c_mask = np.ones(IMG_SHAPE)
        d_mask = d_level.call(depth)
        etype = 2

    else:
        # NOTHING
        c_mask = np.zeros(IMG_SHAPE)
        d_mask = np.zeros(IMG_SHAPE)
        valid = False
        etype = 0
    
    e_mask = e_level.call(c_mask, d_mask)
    f_mask, point, radius = f_level.call(e_mask)

    if valid == True and etype != 0:
        if MIN_DIST <= hag_dist and hag_dist <= MAX_DIST:
            radius_m = px2m(radius, hag_dist)
    else:
        radius_m = 0.0

    return f_mask, point, radius_m, hag_dist, valid, etype

