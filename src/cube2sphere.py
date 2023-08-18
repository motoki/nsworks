from equilib import cube2equi
from PIL import Image

from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cnf: DictConfig):
    img_path = r"D:\reps\nsworks\data\images\horizon_000001.jpg"
    w = 10560
    h = 5280
    mode = "bicubic"
    out_path = "./test.jpg"

    horizon_img = Image.open(img_path)

    # cube2equi
    # * cubemap
    # * cube_format: str dice or horizon
    # * height: int
    # * width : int
    # * mode: str "bilinear" or "bicubic" or "nearest"
    horizon_dat = np.array(horizon_img)
    horizon_dat = horizon_dat.transpose(2,0,1)
    print(horizon_dat.shape)
    sphere = cube2equi(horizon_dat, cube_format="horizon", height=h, width=w, mode=mode)
    sphere = sphere.transpose(1, 2, 0)
    sphere_img = Image.fromarray(sphere)

    sphere_img.save(out_path)

if __name__ == "__main__":
    main()