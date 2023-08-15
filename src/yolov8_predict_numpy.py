from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

def find_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
        
    return None

def main():
    img_path = "./data/images/000001.jpg"
    img = Image.open(img_path)
    img_np = np.asarray(img)

    weight = "./data/weights/yolov8n.pt"
    model = YOLO(weight)

    results = model.predict(img_np, save=True)
    
    img_pil = Image.fromarray(img_np)

    for i, result in enumerate(results):
        print(f"{i}")
        names = result.names
        
        d = ImageDraw.Draw(img_pil)
        for i, box in enumerate(result.boxes):
            cls = box.cls.item()
            name = names[cls]
            # x = box.xywh[0][0].item()
            # y = box.xywh[0][1].item()
            # w = box.xywh[0][2].item()
            # h = box.xywh[0][3].item()

            d.rectangle(box.xyxy.to('cpu').detach().numpy(), outline="blue", width=5)

    out_path = "./data/results/out.jpg"
    img_pil.save(out_path) 

    



if __name__ == "__main__":
    main()

