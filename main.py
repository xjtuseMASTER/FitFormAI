import cv2
from ultralytics import YOLO
from PIL import Image

# 加载模型
model = YOLO("yolov8x-pose.pt")

# 视频处理
def process_video(input_path, output_path, model, conf=0.8):
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
    results = model(source=input_path, stream=True)  # generator of Results objects
    frame_idx = 0
    for r in results:

        keypoints = r.keypoints
        for person in keypoints:
            for x, y, confidence in person:
                pass
            
        # 将结果绘制在帧上
        annotated_frame = r.plot()

        # 写入帧到输出视频
        out.write(annotated_frame)

        frame_idx += 1
        print(f"Processed frame {frame_idx}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 调用函数处理视频
input_video_path = "vedios/杠铃深蹲.mp4"
output_video_path = "output/杠铃深蹲_output_x.mp4"
process_video(input_video_path, output_video_path, model, conf=0.8)