# 后处理模块：将 raw model 输出解析为 DetectionResult 列表
# 并包含：触发条件(area/any)、连续帧计数、历史过滤、报警判定逻辑
# 保持你原有逻辑与字段名一致

from typing import Dict, List, Tuple
import time
import threading

from .types import DetectionResult, DetectionHistory, DetectionModel
from model.email_sender import EmailSender

class PostProcessor:
    """
    后处理器负责：
      - 将 raw_results 转化为 DetectionResult 列表
      - 应用 trigger_mode（area / any）
      - 连续帧计数
      - 历史检测过滤（DetectionHistory）
      - 触发报警的判定（返回是否需要报警及相关信息）
    """
    def __init__(self, models: Dict[str, DetectionModel], area_boxes: List[List[float]] = None, alert_email: str = None):
        self.models = models
        self.enabled_models = {k: True for k in models.keys()}
        self.consecutive_danger_frames = {k: 0 for k in models.keys()}
        self.detection_histories: Dict[str, DetectionHistory] = {}
        self.area_boxes = area_boxes or []
        self.alert_email = alert_email
        self.email_sender = EmailSender()

    def set_area_boxes(self, boxes: List[List[float]]):
        self.area_boxes = boxes

    def set_model_enabled(self, name: str, enabled: bool):
        if name in self.enabled_models:
            self.enabled_models[name] = enabled
            if not enabled:
                self.consecutive_danger_frames[name] = 0

    def parse_raw_results(self, raw_results: Dict[str, object]) -> Dict[str, List[DetectionResult]]:
        """把 raw model 返回值（ultralytics）解析为 DetectionResult"""
        structured = {}
        for name, model_cfg in self.models.items():
            structured[name] = []
            y = raw_results.get(name)
            if y is None:
                continue
            try:
                boxes = y.boxes.xyxy.cpu().numpy() if hasattr(y.boxes, "xyxy") else []
                confs = y.boxes.conf.cpu().numpy() if hasattr(y.boxes, "conf") else []
                cls_ids = y.boxes.cls.cpu().numpy() if hasattr(y.boxes, "cls") else []
                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    cls_name = y.names[int(cls_id)]
                    if cls_name in model_cfg.target_classes and conf >= model_cfg.conf_threshold:
                        structured[name].append(DetectionResult(model_cfg.name, cls_name, float(conf), box.tolist()))
            except Exception:
                continue
        return structured

    def _box_fully_contains(self, container_box: List[float], inner_box: List[float]) -> bool:
        x1, y1, x2, y2 = [float(x) for x in container_box]
        x3, y3, x4, y4 = [float(x) for x in inner_box]
        return (x3 >= x1) and (y3 >= y1) and (x4 <= x2) and (y4 <= y2)

    def update_state_and_check_alert(self, structured_results: Dict[str, List[DetectionResult]]) -> Tuple[bool, Dict]:
        """
        更新内部计数/历史并判断是否需要报警
        返回 (need_alert, info_dict)
        info_dict 包含 danger_detected, danger_boxes, new_detections, alert_parts
        """
        now = time.time()
        danger_detected = {k: False for k in self.models.keys()}
        danger_boxes = {k: [] for k in self.models.keys()}

        # 应用触发模式
        for name, model_cfg in self.models.items():
            if not self.enabled_models.get(name, False):
                continue
            dets = structured_results.get(name, []) or []
            if model_cfg.trigger_mode == 'area':
                for det in dets:
                    for area in self.area_boxes:
                        if self._box_fully_contains(area, det.bbox):
                            danger_detected[name] = True
                            danger_boxes[name].append(det)
                            break
            else:
                if len(dets) > 0:
                    danger_detected[name] = True
                    danger_boxes[name].extend(dets)

        # 更新连续计数
        for name in self.models.keys():
            if not self.enabled_models.get(name, False):
                self.consecutive_danger_frames[name] = 0
                continue
            if danger_detected.get(name, False):
                self.consecutive_danger_frames[name] += 1
            else:
                self.consecutive_danger_frames[name] = 0

        # 判断是否达到报警阈值
        should_alert = False
        alert_parts = []
        for name, model_cfg in self.models.items():
            if not self.enabled_models.get(name, False):
                continue
            if self.consecutive_danger_frames.get(name, 0) >= model_cfg.frame_threshold:
                should_alert = True
                if name == 'glove':
                    alert_parts.append("未佩戴手套")
                elif name == 'head':
                    alert_parts.append("摸头动作")
                else:
                    alert_parts.append(name)

        # 基于历史记录检查是否为新检测
        need_new_alert = False
        new_detections = {}
        if should_alert:
            all_danger = []
            for dets in danger_boxes.values():
                all_danger.extend(dets)
            for det in all_danger:
                cls = det.class_name
                if cls not in self.detection_histories:
                    self.detection_histories[cls] = DetectionHistory()
                if self.detection_histories[cls].is_new_detection(det.bbox, scale_factor=2.0):
                    need_new_alert = True
                    new_detections.setdefault(cls, []).append(det)
            if need_new_alert:
                for cls, dets in new_detections.items():
                    for d in dets:
                        self.detection_histories[cls].add_box(d.bbox)

        info = {
            'danger_detected': danger_detected,
            'danger_boxes': danger_boxes,
            'new_detections': new_detections,
            'alert_parts': alert_parts,
            'all_results': structured_results
        }
        return need_new_alert, info

    def send_alert_async(self, video_name, msg, annotated_frame, original_frame, email=None):
        """异步发送报警邮件（保持原逻辑）"""
        def _send():
            try:
                self.email_sender.send_alert_email(video_name, msg, annotated_frame, email, original_frame, None)
            except Exception:
                pass
        th = threading.Thread(target=_send, daemon=True)
        th.start()
