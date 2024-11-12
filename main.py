import os
from model.setup_model import setup_model
from tasks.set_ups import SetUp
from tasks.task_processor import TaskProcessor


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

# process_by_view("resource/仰卧起坐/侧面视角")



# src_file = "resource/仰卧起坐/侧面视角/肩胛骨未触垫/仰卧起坐-侧面视角-肩胛骨未触垫.MOV"
# task_processor = TaskProcessor(setup_model())
# dest_path = task_processor.process_video2csv(src_file)


setup = SetUp(setup_model())
results = setup.do_analysis("resource/仰卧起坐/侧面视角/腰部弹震借力/仰卧起坐-侧面视角-腰部弹震借力(2).MOV")
print(results)
