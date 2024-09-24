# FitFormAI
基于yolo的健身姿势检测与姿态矫正建议系统

## 项目结构
.
├── README.md
├── config.yaml
├── requirements.txt
├── math_calcu
│   └── angle.py
├── model
│   ├── checkpoints
│   └── main.py
├── tasks
│   └── pull_up.py
├── output
└── vedios

* model:模型方法，checkpoints存储模型参数，main.py目前包含视频处理方法
* tasks:任务文件夹，对于不同的健身任务，分别实现方法
* math_calcu:同用的数学计算工具类
* config.yaml:配置文件
* output:视频输出
* vedios:测试用原始视频