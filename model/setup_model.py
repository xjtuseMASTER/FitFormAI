import os
import yaml
from ultralytics import YOLO


def setup_model() -> YOLO:
    """根据配置文件初始化YOLO模型

    Returns:
        YOLO:完成初始化的YOLO模型
    """
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_FILE = os.path.join(ROOT_DIR, "config.yaml")

    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    CHECKPOINTS = config["CHECKPOINTS"]
    MODELNAME = config["MODELNAME"]
    return YOLO(ROOT_DIR+ '/' + CHECKPOINTS + MODELNAME)
