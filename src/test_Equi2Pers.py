import os
import cv2
import time
import argparse
import warnings
from typing import Any, Callable, Dict
from copy import deepcopy
import numpy as np
from PIL import Image
from equilib import Equi2Cube

from functools import partial, update_wrapper

SAVE_ROOT="./data/results"
DATA_ROOT="./data/images"
IMG_NAME = "000001.jpg"






def test_equi2Cube(image_path):

    pass


def wrapped_partial(func: Callable[[], Any], *args, **kwargs) -> Callable[[], Any]:
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func
    

def printable_time(name: str, time: float, prec: int = 6) -> str:
    return f"Func: {name}\t" + format(time, f".{str(prec)}f")

def func_timer(func: Callable[[], Any], *, ret: bool = False,
                verbose:bool = True, prec: int = 6) -> Callable[[], Any]:
    
    def wrap(*args, **kwargs):

        # time.timeよりtime.pref_counter()のほうが正確に時間が計れる
        tic = time.pref_counter()
        results = func(*args, **kwargs)
        toc = time.pref_counter()

        if verbose:
            print(printable_time())
        if ret:
            return results, toc - tic
        
        return results
    
    return wrap


def _open_as_PIL(img_path: str) -> Image.Image:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    img = Image.open(img_path)
    assert img is not None
    if img.getbands() == tuple("RGBA"):
        # NOTE: Sometimes images are RGBA
        img = img.convert("RGB")
    return img


def _open_as_cv2(img_path: str) -> np.ndarray:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    # FIXME: shouldn't use `imread` since it won't auto detect color space
    warnings.warn("Cannot handle color spaces other than RGB")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    assert img is not None
    return img


def load2numpy(img_path: str, dtype: np.dtype, is_cv2: bool = False) -> np.ndarray:

    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    if is_cv2:
        # FIXME: currently only supports RGB
        img = _open_as_cv2(img_path)
    else:
        img = _open_as_PIL(img_path)
        img = np.asarray(img)

    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    img = np.transpose(img, (2, 0, 1))

    # NOTE: Convert dtypes
    # if uint8, keep 0-255
    # if float, convert to 0.0-1.0
    dist_dtype = np.dtype(dtype)
    if dist_dtype in (np.float32, np.float64):
        img = img / 255.0
    img = img.astype(dist_dtype)

    return img

def get_image(dtype: np.dtype = np.dtype(np.float32)):
    path = os.path.join(DATA_ROOT, IMG_NAME)
    img = load2numpy(path, dtype=dtype, is_cv2=False)
    return img

def make_batch(img: np.ndarray, bs: int = 1):
    imgs = np.empty((bs, *img.shape), dtype=img.dtype)
    for b in range(bs):
        imgs[b, ...] = deepcopy(img)
    return imgs


def bench_baselines(bs:int, height:int, width:int, mode:str, dtype:np.dtype=np.dtype(np.float32), save_output:bool = False):

    if dtype == np.float32:
        rtol = 1e-03
        atol = 1e-05
    elif dtype == np.float64:
        rtol = 1e-05
        atol = 1e-08
    else:
        rtol = 1e-01
        atol = 1e-03

    img = get_image(dtype)

    imgs = make_batch(img, bs=bs)

    print("Batch is maded")
    print(imgs.shape, img.dtype)

    print("Scipy")
out_scipy = func_timer(run_scipy)(

)


run_cv2 = wrapped_partial(run, override_func = grid_sample_cv2)
run_scipy = wrapped_partial(run, override_func =)

def main():
    image_path = r"C:\Users\skyfl\reps\nsworks\data\images\000001.jpg"

    save_outputs = True

    bs = 8
    height=5280
    width=10560
    dtype = np.dtype(np.float32)
    mode = "bilinear"

    bench_baselines(
        bs=bs,
        height=height,
        width = width,
        mode = mode,
        dtype = dtype,
        save_outputs = save_outputs
    )




if __name__ == "__main__":
    main()