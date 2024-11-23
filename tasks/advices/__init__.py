import os
import json

def load_advice_by_filename( filename: str):
    """根据文件名从指定文件夹加载 JSON 文件并返回其内容"""
    file_path = os.path.join('advices', filename)
    
    if not os.path.isfile(file_path):
        print(f"错误：文件 {filename} 不存在于 tasks/advices 中。")
        return None
    
    if not filename.endswith('.json'):
        print(f"错误：文件 {filename} 不是一个 JSON 文件。")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data  
    except json.JSONDecodeError:
        print(f"错误：文件 {filename} 无法解析为 JSON 格式。")
        return None
