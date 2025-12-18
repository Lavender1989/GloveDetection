# 区域加载器：调用view2xml，将区域做校验与裁剪后返回
from typing import List, Tuple
import os

def load_area_for_view(view_index: int, width: int, height: int):
    """
    调用 view2xml(view_index, width, height) 并返回 (valid_area_boxes, log_message)
    对区域做基本校验与裁剪，保证所有坐标在 [0,width]x[0,height] 范围内，且 x1<x2, y1<y2
    """
    try:
        from controller.video_view_mapping import view2xml
    except Exception as e:
        return [], f"无法导入 view2xml: {e}"

    try:
        area_boxes, log_msg = view2xml(view_index, width, height)
    except Exception as e:
        return [], f"调用 view2xml 失败: {e}"

    valid_boxes = []
    for i, box in enumerate(area_boxes):
        try:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(x) for x in box]
            # Normalize / clip
            if width and height:
                x1 = max(0, min(x1, width))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height))
                y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            valid_boxes.append([x1, y1, x2, y2])
        except Exception:
            continue

    return valid_boxes, log_msg
