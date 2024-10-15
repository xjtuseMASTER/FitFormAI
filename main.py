import sys
sys.path.append("./tasks")
import yaml
from pathlib import Path
from ultralytics import YOLO
from tasks.task_processor import TaskProcessor


def setup_model() -> YOLO:
    """根据配置文件初始化YOLO模型

    Returns:
        YOLO:完成初始化的YOLO模型
    """
    CONFIG_FILE = "config.yaml"
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    CHECKPOINTS = config["CHECKPOINTS"]
    MODELNAME = config["MODELNAME"]
    return YOLO(CHECKPOINTS + MODELNAME)


src_file = "./resource/引体向上/背部视角/标准/引体向上-背部-标准-01.mov"
task_processor = TaskProcessor(setup_model())
dest_path = task_processor.process_video2vedio(src_file)