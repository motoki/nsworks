import os
from io import BytesIO
from PIL import Image,ExifTags, ImageDraw
from omegaconf import DictConfig, OmegaConf
import hydra
from ultralytics import YOLO
import matplotlib.pyplot as plt

from equilib import equi2cube, cube2equi
import numpy as np
import torch

from tqdm import tqdm

def image_processor(image: Image):
    # Check if the image has Exif information

    image_dat = np.array(image)
    image_dat = image_dat.transpose(1, 2, 0)

    rots = {
        "roll": 0,
        "pitch": 0,
        "yaw": 0
    }

    cube_data = equi2cube(image_dat, rots=rots, w_face=3368, cube_format="horizon")
    cube_data = cube_data.transpose(2, 1, 0)
    cube_img = Image.fromarray(cube_data)

    return cube_img
    

def drawRects(image, results, colors):
    img_pil = Image.fromarray(image[:, :, ::-1])

    for i, result in enumerate(results):
        names = result.names

        draw = ImageDraw.Draw(img_pil)
        for i, box in enumerate(result.boxes):
            cls = int(box.cls.item())
            c = colors[cls]
            name = names[cls]
            draw.rectangle(box.xyxy.to('cpu').detach().numpy(), outline=c, width=5)
    
    return img_pil

def process_image(image: Image, rots, w_face, height, width, model, colors) -> Image:
    ##############
    # numpy に変換
    image_dat = np.array(image)
    image_dat = image_dat.transpose(2, 0, 1)

    ##################
    #horizon形式に変換'
    image_dat = equi2cube(image_dat, rots=rots, w_face=w_face, cube_format="horizon")

    #前後左右上下ごとに物体検出を行う
    image_dat = image_dat.transpose(1, 2, 0)

    #image_pil = Image.fromarray(image_dat)

    image_slices = []
    for i in range(6):
        tmp_dat = np.ascontiguousarray(image_dat[:, i*w_face:(i+1)*w_face,::-1])
        results = model.predict(tmp_dat, save=False, imgsz=w_face, conf=0.2)
        
        # 結果を描画
        tmp_dat_pil = drawRects(tmp_dat, results, colors)
        image_slices.append((tmp_dat_pil, results))
    
    # 描画した画像を結合
    horizon_img = Image.new('RGB', (w_face*6, w_face))

    offset = 0
    for img in image_slices:
        horizon_img.paste(img[0], (offset, 0))    
        offset += w_face

    horizon_img_dat = np.array(horizon_img)
    horizon_img_dat = horizon_img_dat.transpose(2, 0, 1)

    #######################
    #eqirectanglar形式に変換'
    eqirect_dat = cube2equi(horizon_img_dat, cube_format="horizon", height=height, width=width)
    # PIL Imageに変換
    eqirect_dat = eqirect_dat.transpose(1, 2, 0)
    return Image.fromarray(eqirect_dat)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig):

    # 設定値の読み込み
    images_dir = cfg.data.images
    out_dir = cfg.data.results.output
    weight = cfg.data.yolo.weight
    w_face = cfg.data.w_face    #w_face = 3392 #3368

    rots = {
        "roll": cfg.data.rots.roll,
        "pitch": cfg.data.rots.pitch,
        "yaw": cfg.data.rots.yaw
    }

    # rots = {
    #     "roll": 0,
    #     "pitch": 0,
    #     "yaw" : 0,
    # }

    colors = cfg.data.colors

    # 出力先フォルダの用意
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # フォルダから画像ファイルを抽出
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for root, _, files in os.walk(images_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions ):
                image_files.append(os.path.join(root, file))

    progress_bar = tqdm(total=len(image_files), desc='Process Images', unit='image')

    # model 
    model = YOLO(weight)

    for image_file in image_files:
        # process image
        img = Image.open(image_file)

        processed_img = process_image(img, rots, w_face, img.height, img.width, model, colors)

        # 出力先フォルダの作成
        out_path = image_file.replace(images_dir, out_dir)
        parent_dir = os.path.dirname(out_path)

        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # copy exif and save
        exif = img.info['exif']
        if exif:
            processed_img.save(out_path, exif=exif)
        else:
            processed_img.save(out_path)


        progress_bar.update(1)
    progress_bar.close()
        

    # image folder から画像を順次読み込む


        # 360 の場合
        ## 360画像をCubeマップへ変換方式はhorizon形状
        
        ## horizon形式を順に切り分けて物体検出にかける
            ## 検出結果を描画

        
        ## 描画が終わった画像を再度horizon形式に

        ## horizon形式からSphere形式に変換

        ## exifをコピー

        ## 画像を保存


        # 非360の場合
        ## 物体検出
        
        ## 検出結果を描画

        ## 位置情報をコピー

        ## 検出結果を保存


if __name__ == "__main__":
    main()