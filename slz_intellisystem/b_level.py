import numpy as np

from params import *
import c_level
import d_level
import e_level
import f_level


def px2m(r:float, h: float):
    delta = np.mean([
        np.deg2rad(C_FOV[0] / 2) * h / IMG_SHAPE[0],
        np.deg2rad(C_FOV[1] / 2) * h / IMG_SHAPE[1]
    ])
    return r * delta


def disp(model, color, depth, altitude):
    valid = False
    etype = 0
    c_mask = np.zeros(IMG_SHAPE)
    d_mask = np.zeros(IMG_SHAPE)

    if HIGH_DIST < altitude and altitude <= MAX_DIST:
        # COLOR-only
        c_mask = c_level.call(model, color)
        d_mask = np.ones(IMG_SHAPE)
        valid = True
        etype = 1

    elif LOW_DIST < altitude and altitude <= HIGH_DIST:
        # COLOR & DEPTH
        c_mask = c_level.call(model, color)
        d_mask = d_level.call(depth)
        valid = True
        etype = 2

    elif MIN_DIST <= altitude and altitude <= LOW_DIST:
        # DEPTH-only
        c_mask = np.ones(IMG_SHAPE)
        d_mask = d_level.call(depth)
        valid = True
        etype = 3

    else:
        # NOTHING
        c_mask = np.zeros(IMG_SHAPE)
        d_mask = np.zeros(IMG_SHAPE)
        valid = False
        etype = 0
    
    e_mask = e_level.call(c_mask, d_mask)
    f_mask, x_coord, y_coord, radius = f_level.call(e_mask)

    if valid == True and etype != 0:
        if etype in [2, 3]:
            radius_m = px2m(radius, altitude)
    else:
        radius_m = 0.0

    return e_mask, f_mask, x_coord, y_coord, radius_m, valid, etype

