import os
import sys
from typing import List, Tuple

from scipy.fft import fft, ifft

sys.path.append("./tasks")
from matplotlib import pyplot as plt
import yaml
from pathlib import Path
from ultralytics import YOLO
from tasks.task_processor import TaskProcessor
from tasks.set_ups import SetUp
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


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



def process_by_view(src_view_dir: str) -> None:
    """按文件夹结构,将resource文件夹中所有样本都进行数据提取,并输出为cSV文件到output文件夹中方便下一步绘图"""
    dest_dir = 'output'
    task_processor = TaskProcessor(setup_model())

    for root, dirs, files in os.walk(src_view_dir):
        for file in files:
            if file == '.gitkeep': continue 
            if file.endswith(('.MOV', '.mov', '.mp4')):
                src_path = os.path.join(root, file)
                # processing
                task_processor.process_video2csv(src_path)

if __name__ == "__main__":
    process_by_view("resource/俯卧撑/正面视角")

# src_file = "resource/仰卧起坐/侧面视角/肩胛骨未触垫/仰卧起坐-侧面视角-肩胛骨未触垫.MOV"
# task_processor = TaskProcessor(setup_model())
# dest_path = task_processor.process_video2csv(src_file)


# setup = SetUp(setup_model())
# results = setup.do_analysis("resource/仰卧起坐/侧面视角/腰部弹震借力/仰卧起坐-侧面视角-腰部弹震借力(2).MOV")
# print(results)
