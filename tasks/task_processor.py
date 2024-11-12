import os
from . import pull_up, set_ups
from pathlib import Path
from ultralytics import YOLO

class TaskProcessor:
    """对YOLO模型以及任务处理方法的封装,按照resource文件夹目录结构,通过文件路径获得相应的数据提取方法,以及相应的输出路径
    """
    def __init__(self, model: YOLO) -> None:
        self.model = model
        self.video2csv_methods = {  
            #同一视角可以提取的数据特征基本一致，故video2csv_methods以**动作/视角**区分
            '引体向上/背部视角':pull_up.back_video2csv,
            '引体向上/侧面视角':pull_up.side_video2csv,
            '仰卧起坐/侧面视角':set_ups.side_video2csv
        }
        self.video2video_methods = {
            '引体向上/背部视角':pull_up.back_video2video,
            '引体向上/侧面视角':pull_up.side_video2video,
            '仰卧起坐/侧面视角':set_ups.side_video2video
        }
    

    def process_video2csv(self, input_path: str, **keywarg: any) -> str:
        """处理视频特征并输出为csv文件的统一方法，通过input_path判断任务处理为csv的方法，并调用相关函数.该方法统一传入参数input_path

        Args:
            input_path (str): 输入视频路径，例如："resource/引体向上/背部视角/标准/引体向上-背部-标准-01.mov"

        Returns:
            str: 输出csv文件的路径
        """
        task = '/'.join(Path(input_path).parts[1:3])
        output_path = input_path.replace("resource", "output").split('.')[0] + '.csv'

        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            # add .gitkeep
            with open(os.path.join(directory, '.gitkeep'), 'w') as gitkeep:
                gitkeep.write('')

        method = self.video2csv_methods.get(task)
        if method:
            method(input_path,output_path, self.model)
            print(f"Processed {input_path} to {output_path}")
            return output_path
        else:
            raise ValueError(f"错误:未找到该任务的csv处理方法: {task},请查看TaskProcessor类video2csv_methods属性")
        

    
    def process_video2vedio(self, input_path: str, **keywarg: any) -> str:
        """将数据特征标记到每一视频帧的统一方法，通过input_path判断任务处理为视频的方法，并调用相关函数.该方法统一传入参数input_path

        Args:
            input_path (str): 输入视频路径，例如："resource/引体向上/背部视角/标准/引体向上-背部-标准-01.mov"

        Returns:
            str: 输出mp4文件的路径
        """
        task = '/'.join(Path(input_path).parts[1:3])
        output_path = input_path.replace("resource", "output").split('.')[0] + '.mp4'

        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            # add .gitkeep
            with open(os.path.join(directory, '.gitkeep'), 'w') as gitkeep:
                gitkeep.write('')

        method = self.video2video_methods.get(task)
        if method:
            method(input_path, output_path, self.model)
            print(f"Processed {input_path} to {output_path}")
            return output_path
        else:
            raise ValueError(f"错误:未找到该任务的视频帧处理方法: {task},请查看TaskProcessor类video2vedio_methods属性")