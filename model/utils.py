import torch

__all__ = ["extract_main_person"]

def extract_main_person(result: torch.Tensor) -> torch.Tensor:
    """抽取画面中的主体人物，一般来说拍摄主体总会是面积最大的一个，所以可以通过比较所有person的面积得到主体人物

    Args:
        result (torch.Tensor): yolo返回的results中的一帧

    Returns:
        (torch.Tensor): 主体人物的keypoints
    """
    boxes = result.boxes
    max_area = 0
    main_person_index = 0
    person_class_id = 0  #用于检测类别是否为Person

    for i, (x, y, w, h) in enumerate(boxes.xywh):
        area = w * h
        if boxes.cls[i] == person_class_id and area > max_area:
            max_area = area
            main_person_index = i

    return result.keypoints[main_person_index].data.squeeze()