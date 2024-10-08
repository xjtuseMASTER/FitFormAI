import sys
sys.path.append("./tasks")

import cv2
import yaml
import pandas as pd
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

    # # 定义视频写入对象
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编解码器
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 处理视频帧
    results = model(source=input_path, stream=True, **keywarg)  # generator of Results objects
    frame_idx = 0
    angles_data = []
    for r in results:
        # 将结果绘制在帧上
        annotated_frame = r.plot()

        keypoints = extract_main_person(r)
        # processing
        angles = pull_up.process_angle(keypoints)
        angles_data.append(angles)

        # out.write(annotated_frame)
            
        frame_idx += 1
        print(f"Processed frame {frame_idx}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    pull_up.plot_angles(angles_data, frame_idx)
    df = pd.DataFrame(angles_data, columns=['左肘角度', '右肘角度', '左肩角度', '右肩角度'], index= list(range(1, frame_idx + 1)))
    df.to_csv(output_video_path, index_label='帧索引')

    # 释放资源
    cap.release()
    # out.release()
    cv2.destroyAllWindows()



# 调用函数处理视频
input_video_path = "resource/引体向上/背部视角/标准/引体向上-背部-标准-01.mov"
output_video_path = "output/引体向上/背部视角/标准/引体向上-背部-标准-01.csv"
process_video(input_video_path, output_video_path, model, conf=0.8)