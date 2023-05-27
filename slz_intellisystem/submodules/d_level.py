import numpy as np

from .params import IMG_SHAPE, W_SIZE, E_AREA, E_STD


def decomp_image(d_im: np.ndarray) -> np.ndarray:
    d_map = np.empty(
        shape=(d_im.shape[0] // W_SIZE, d_im.shape[1] // W_SIZE),
        dtype=list
    )

    for r in range(d_map.shape[0]):
        for c in range(d_map.shape[1]):
            d_map[r, c] = d_im[
                r*W_SIZE : (r+1)*W_SIZE,
                c*W_SIZE : (c+1)*W_SIZE
            ]
    
    return d_map


def calc_stat(im: np.ndarray) -> np.ndarray:
    loc_list = np.array([p for p in im.flatten() if p > 0.0])

    size = loc_list.size
    area = size / loc_list.shape[0] if size > 0 else 0
    std = np.std(loc_list) if size > 0 else 0
    average = np.average(loc_list) if size > 0 else 0
    median = np.median(loc_list) if size > 0 else 0

    fl = 0
    if median != 0:
        if median > (H_MEDIAN + E_STD):
            fl = 1
        else:
            fl = 2
    else:
        fl = 0

    res = fl if area >= E_AREA and std <= E_STD and fl != 0 else 0
    return res


def calc_w(comp_im) -> np.ndarray:
    loc_list = np.zeros(comp_im.shape)

    for i, arr in enumerate(comp_im.flatten()):
        loc_list[i // loc_list.shape[1], i % loc_list.shape[1]] = \
            calc_stat(arr)
        
    count1 = np.count_nonzero(loc_list.flatten() == 1)
    count2 = np.count_nonzero(loc_list.flatten() == 2)
    if count2 >= count1:
        loc_list = np.where((loc_list == 2), 1, 0)
    else:
        loc_list = np.where((loc_list == 1), 1, 0)

    return loc_list


def get_mask(comp_im) -> np.ndarray:
    mask = np.zeros(IMG_SHAPE, dtype=np.uint8)

    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            mask[r, c] = 1 if comp_im[r // W_SIZE, c // W_SIZE] else 0

    return mask


def call(depth: np.ndarray) -> np.ndarray:
    global H_AVERAGE
    global H_MEDIAN

    H_AVERAGE = np.average(depth.flatten())
    H_MEDIAN = np.median(depth.flatten())

    d_map = decomp_image(depth)
    f_list = calc_w(d_map)
    mask = get_mask(f_list)

    return mask

