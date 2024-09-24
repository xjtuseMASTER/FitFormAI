import cv2
import yaml
import torch
from ultralytics import YOLO
from tasks import pull_up

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHECKPOINTS = config["CHECKPOINTS"]
MODELNAME = config["MODELNAME"]

# 加载模型
model = YOLO(CHECKPOINTS + MODELNAME)

# 视频处理
def process_video(input_path: str, output_path: str, model: YOLO, **keywarg: any) -> None:
    """
    使用yolo处理视频,视频类型为mp4

    Args:
        input_path (str): 输入视频地址
        output_path (str): 输出视频地址
        model (YOLO): 所使用的YOLO模型
    """
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频的宽度、高度和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 定义视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编解码器
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 处理视频帧
    results = model(source=input_path, stream=True, **keywarg)  # generator of Results objects
    return results,out,cap