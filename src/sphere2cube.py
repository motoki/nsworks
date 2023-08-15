import os
import numpy as np
import warnings
from equilib import equi2cube
import cv2

from PIL import Image


def main():
    image_path = "./data/images/000001.jpg"

    equi_img = Image.open(image_path)
    mode = equi_img.mode

    equi_img = np.asarray(equi_img)
    print("equi_img: ", equi_img.shape)

    equi_img = np.transpose(equi_img, (2, 0, 1))

    print(equi_img.shape)

    rots = {
        "roll": 0,
        "pitch": 0,
        "yaw": 0
    }

    mode = "bicubic"
    # mode  = "nearest"
    cube = equi2cube(equi = equi_img,cube_format="horizon", rots=rots, w_face=2640, z_down=False, mode=mode)
    
    print("cube.shape", cube.shape)
    print("type: ", type(cube))
    print("size: ", cube.size, "shape: ", cube.shape)
    cube = cube.transpose(1,2,0)
    out_image = Image.fromarray(cube, mode)

    out_path = "./data/results/00001.jpg"
    out_image.save(out_path)

if __name__ == "__main__":
    main()