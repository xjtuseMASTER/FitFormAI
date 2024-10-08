import sys
sys.path.append("./tasks")

import os
import cv2
import yaml
from pathlib import Path
import pandas as pd
from tasks import pull_up
from ultralytics import YOLO
from tasks.task_processor import TaskProcessor

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHECKPOINTS = config["CHECKPOINTS"]
MODELNAME = config["MODELNAME"]

# 加载模型
model = YOLO(CHECKPOINTS + MODELNAME)


def resource2output() -> None:
    """按文件夹结构,将resource文件夹中所有样本都进行数据提取,并输出为csv文件到output文件夹中方便下一步绘图,output文件夹结构和resource一致
    """
    src_dir = "resource"
    dest_dir = 'output'
    task_processor = TaskProcessor()
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file == '.gitkeep': continue
            if file.endswith(('.mov', '.mp4')):  
                src_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_path, src_dir)
                relative_path_csv = relative_path.split('.')[0] + '.csv'
                dest_path = os.path.join(dest_dir, relative_path_csv)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                task = '/'.join(Path(src_path).parts[1:4])
                if task in task_processor.task_process_methods.keys():
                    task_processor.process_task(src_path,dest_path,model)

                print(f"Processed {src_path} to {dest_path}")


resource2output()



