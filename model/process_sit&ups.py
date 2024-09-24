import cv2
import yaml
import torch
from ultralytics import YOLO
from tasks import pull_up
import utils

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
    results,out,cap=utils.process_video(input_path,output_path,model)
    frame_idx = 0
    for r in results:

        # 将结果绘制在帧上
        annotated_frame = r.plot()

        keypoints = r.keypoints
        for person in keypoints:
            annotated_frame = pull_up.process_angle(annotated_frame, person.data.squeeze(0))
            points = person.data.squeeze(0)
            # for i in range(points.size(0)):
            #     print(points[i])
            
        # 写入帧到输出视频
        out.write(annotated_frame)

        frame_idx += 1
        print(f"Processed frame {frame_idx}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 调用函数处理视频
input_video_path = "vedios/引体向上.mp4"
output_video_path = "output/引体向上_output_x.mp4"
process_video(input_video_path, output_video_path, model, conf=0.8)