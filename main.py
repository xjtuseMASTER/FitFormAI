import sys
sys.path.append("./tasks")

import cv2
import yaml
from utils import extract_main_person
from tasks import pull_up
from ultralytics import YOLO

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
    frame_idx = 0
    for r in results:
        # 将结果绘制在帧上
        annotated_frame = r.plot()

        keypoints = extract_main_person(r)
        # processing
        # annotated_frame = pull_up.process_angle(annotated_frame, keypoints)
        # annotated_frame = pull_up.is_wrist_above_elbow(annotated_frame, keypoints)  
        annotated_frame = pull_up.show_keypoints(annotated_frame, keypoints)

        out.write(annotated_frame)
            
        frame_idx += 1
        print(f"Processed frame {frame_idx}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()



# 调用函数处理视频
input_video_path = "vedios/引体向上_3.MP4"
output_video_path = "output/引体向上_3.mp4"
process_video(input_video_path, output_video_path, model, conf=0.8)