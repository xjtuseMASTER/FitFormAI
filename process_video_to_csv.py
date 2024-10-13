import sys
sys.path.append("./tasks")
import yaml
from pathlib import Path
from ultralytics import YOLO
from tasks.task_processor import TaskProcessor


def setup_model(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    CHECKPOINTS = config["CHECKPOINTS"]
    MODELNAME = config["MODELNAME"]
    return YOLO(CHECKPOINTS + MODELNAME)


def video2output_(src_file: str, model) -> None:
    task_processor = TaskProcessor()
    dest_path = src_file.replace("resource", "output").split('.')[0] + '.csv'
    task_processor.process_task(src_file, dest_path, model)
    print(f"Processed {src_file} to {dest_path}")
    
    
def video2output(src_file:str) -> None:
    model = setup_model("config.yaml")
    video2output_(src_file, model)

video2output("./resource/引体向上/背部视角/标准/引体向上-背部-标准-01.mov")
    