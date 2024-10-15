import pull_up
from pathlib import Path
from ultralytics import YOLO

class TaskProcessor:
    """对任务处理方法的包装,按照resource文件夹目录结构,通过文件路径获得相应的数据提取方法
    """
    def __init__(self) -> None:
        self.task_process_methods = {
            # '引体向上/背部视角/标准':pull_up.processor_standard,
            '引体向上/背部视角/脊柱侧弯':pull_up.processor_standard,
            # '引体向上/背部视角/肩胛不稳定':pull_up.processor_standard,
            # '引体向上/背部视角/握距不合适':pull_up.processor_standard
        }
    
    def process_task(self, input_path: str, output_path: str, model: YOLO, **keywarg: any) -> None:
        """动作分析统一方法，通过input_path判断任务处理方法，并调用相关函数.该方法统一传入参数input_path、output_path、model

        Args:
            input_path (str): 输入视频路径
            output_path (str): 输出路径
            model (YOLO): 所使用的YOLO模型

        """
        task = '/'.join(Path(input_path).parts[1:4])
        method = self.task_process_methods.get(task)
        if method:
            return method(input_path,output_path,model)
        else:
            raise ValueError(f"错误:未找到任务的处理方法: {task}")