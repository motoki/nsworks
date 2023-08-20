from ultralytics import YOLO
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="config", config_name="pose-config")
def main(cfg: DictConfig):
    print(cfg)
    weight = cfg.yolo.weight
    movie = cfg.yolo.movie

    model = YOLO(weight)
    results = model.predict(movie, save=True)
    print(results)


if __name__ == "__main__":
    main()