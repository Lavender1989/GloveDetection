# 绘制模块：负责把结果画到帧上（区域、检测框、报警横幅等）

import cv2
from typing import Dict, List
from .types import DetectionResult

class Drawer:
    def __init__(self):
        pass

    def draw(self, frame, all_results: Dict[str, List[DetectionResult]], danger_detected: Dict[str, bool], danger_boxes: Dict[str, List[DetectionResult]], area_boxes: List = None, alert_active: bool = False, alert_remaining: float = 0.0):
        """
        在 frame 上绘制：
          - 区域 boxes（黄色）
          - 每个模型的检测结果（不同颜色）
          - 危险时的红色高亮
          - 报警横幅（若 alert_active True）
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        font_scale = 0.9
        line_thickness = max(1, int(font_scale * 2))

        # 绘制检测区域
        if area_boxes:
            for box in area_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # draw detections
        for name, dets in all_results.items():
            color = (255, 0, 0) if name == 'glove' else (0, 255, 0)
            label_prefix = "bare" if name == 'glove' else "touch"
            for det in dets:
                x1, y1, x2, y2 = map(int, det.bbox)
                is_danger = any(d.bbox_equals(det.bbox) for d in danger_boxes.get(name, []))
                if is_danger:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,0,255), max(2, line_thickness+1))
                    cv2.putText(annotated, f"DANGER: {label_prefix} {det.confidence:.2f}", (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                else:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)
                    cv2.putText(annotated, f"{label_prefix} {det.confidence:.2f}", (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

        # alert banner
        if alert_active:
            cv2.putText(annotated, f"ALERT ({alert_remaining:.1f}s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        return annotated
