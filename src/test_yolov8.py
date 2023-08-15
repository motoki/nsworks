from ultralytics import YOLO
from PIL import Image
import pprint 


def main():
    #yaml = "yolov8n.yaml"
    weight = "./data/weights/yolov8n.pt"
    model = YOLO(weight)

    images= ["./data/images/hourse.jpg",
            "./data/images/Cat03.jpg",
            "./data/images/bus.jpg"]

    results = model.predict(images, save=True, conf=0.2)

    for i, result in enumerate(results):
        boxes = result.boxes
        probs = result.probs
        print("-----------------")
        print(f"{i}番目結果")
        print(boxes)
        print(probs)

if __name__ == "__main__":
    main()
