"""
多模型检测工作类 - 支持同时运行多个YOLO模型进行检测
"""
import time
import os
import xml.etree.ElementTree as ET
from typing import Dict, List
import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker
from PyQt6.QtGui import QImage
from ultralytics import YOLO
from uuid import uuid4
from datetime import datetime

from model.email_sender import EmailSender
from model.video_buffer_manager import VideoBufferManager
from controller.video_view_mapping import load_area_from_xml, view2xml


class DetectionResult:
    """单个检测结果"""
    def __init__(self, detection_type: str, class_name: str, confidence: float, bbox: List[float]):
        self.detection_type = detection_type  # 'glove' 或 'head'
        self.class_name = class_name  # 'bare' 或 'touch'
        self.confidence = confidence
        self.bbox = [float(x) for x in bbox]  # [x1, y1, x2, y2]
    def bbox_equals(self, other_bbox: List[float], tol: float = 1e-3) -> bool:
        if other_bbox is None or len(other_bbox) != 4:
            return False
        return all(abs(float(a) - float(b)) <= tol for a, b in zip(self.bbox, other_bbox))


class DetectionModel:
    """包含模型、阈值与触发策略"""
    def __init__(self,
                 name: str,
                 model_path: str,
                 detection_type: str,
                 target_classes: List[str],
                 conf_threshold: float = 0.5,
                 frame_threshold: int = 2,
                 trigger_mode: str = "area"):  # trigger_mode: "area" or "any"
        """
        name: 模型 key, 如 'glove' / 'head'
        trigger_mode:
            - 'area' : 只有 bbox 完全进入 area 区域才计为危险（适合 glove）
            - 'any'  : 只要检测到就计为危险（适合 touch）
        """
        self.name = name
        self.model = YOLO(model_path)
        # 尝试把模型移到 GPU
        try:
            self.model.to('cuda')
        except Exception:
            pass
        self.detection_type = detection_type
        self.target_classes = target_classes
        self.conf_threshold = conf_threshold
        self.frame_threshold = frame_threshold
        self.trigger_mode = trigger_mode
        self.enabled = True  # 可由外部配置开关


class MultiDetectorWorker(QObject):
    """多模型检测工作类"""
    proc_frame_ready = pyqtSignal(QImage)  # 处理后的帧信号
    log_message = pyqtSignal(str)  # 日志信号
    alert_message = pyqtSignal(str)  # 报警信号

    def __init__(self,
                 models_config: Dict[str, Dict],
                 video_name: str,
                 view_index: int,
                 alert_email: str,
                 parent=None):
        """
       models_config 形如：
        {
            'glove': {
                'path': 'xxx.pt',
                'target_classes': ['bare'],
                'conf': 0.8,
                'frame_threshold': 10,
                'trigger_mode': 'area'
            },
            'head': {
                'path': 'yyy.pt',
                'target_classes': ['touch'],
                'conf': 0.8,
                'frame_threshold': 3,
                'trigger_mode': 'any'
            }
        }
        """
        super().__init__(parent)
        self._mutex = QMutex()
        
        # 新增：用于防止重复报警的变量
        self.last_alert_boxes = {}  # 存储上次报警的检测框，格式：{model_name: [bbox]}
        self.last_alert_time = 0  # 上次报警的时间戳

        # 初始化模型字典
        self.models: Dict[str, DetectionModel] = {}
        for name, cfg in models_config.items():
            self.models[name] = DetectionModel(
                name=name,
                model_path=cfg['path'],
                detection_type=cfg.get('detection_type', name),
                target_classes=cfg.get('target_classes', []),
                conf_threshold=cfg.get('conf', 0.5),
                frame_threshold=cfg.get('threshold', cfg.get('frame_threshold', 2)),
                trigger_mode=cfg.get('trigger_mode', 'any')
            )

        # 运行时可通过UI修改哪些模型启用 (key->bool)
        self.enabled_models: Dict[str, bool] = {name: True for name in self.models.keys()}
        
        # 警报与计数
        self.consecutive_danger_frames: Dict[str, int] = {name: 0 for name in self.models.keys()}
        self.alert_active = False
        self.alert_start_time = 0
        self.ALERT_DISPLAY_SECONDS = 5  # UI 可覆盖
        self.show_ui = True

        self.video_name = video_name
        self.view_index = view_index
        self.alert_email = alert_email

        
        # 区域检测相关变量（仅手套检测使用）
        self.area_boxes = []
        # 移动到video_view_mapping中
        # self.view_names = ["视角1", "视角2", "视角3", "视角4", "视角5", "视角6", "视角7", "视角8", "视角9", "视角10"]
        # self.xml_paths = [
        #     os.path.join(os.path.dirname(__file__), "..", "area", "0911_1_frame00000.xml"),  # VIEW_1
        #     os.path.join(os.path.dirname(__file__), "..", "area", "0911_2_frame00000.xml"), # VIEW_2
        #     os.path.join(os.path.dirname(__file__), "..", "area", "301.xml"), # VIEW_3
        #     os.path.join(os.path.dirname(__file__), "..", "area", "401.xml"), # VIEW_4
        #     os.path.join(os.path.dirname(__file__), "..", "area", "501.xml"), # VIEW_5
        #     os.path.join(os.path.dirname(__file__), "..", "area", "601.xml"), # VIEW_6
        #     os.path.join(os.path.dirname(__file__), "..", "area", "701.xml"), # VIEW_7
        #     os.path.join(os.path.dirname(__file__), "..", "area", "901.xml"), # VIEW_8
        #     os.path.join(os.path.dirname(__file__), "..", "area", "1201.xml"), # VIEW_9
        #     os.path.join(os.path.dirname(__file__), "..", "area", "1301.xml"), # VIEW_10
        # ]
        
        # 邮件发送器
        self.email_sender = EmailSender()
        self.processed_alert_frame = None
        
        # 视频缓冲管理器，用于存储最近30秒的视频帧 (但是一直没办法缓存到真实30s)
        self.buffer_dir = os.path.join(os.path.dirname(__file__), '..', 'temp_video_buffer')
        # 确保缓冲目录存在
        os.makedirs(self.buffer_dir, exist_ok=True)
        self.video_buffer = VideoBufferManager(buffer_seconds=30, fps=30)
        
        # 用于保存当前视频的固定路径
        self.current_video_path = os.path.join(self.buffer_dir, f'current_video_{self.video_name.replace(" ", "_")}.mp4')
        
        # 记录最后一次保存视频的时间
        self.last_save_time = time.time()
        
        # 尝试加载对应的XML文件
        self.width = None
        self.height = None
        # self._load_area_for_view()
        # 输出真实初始化的模型
        model_desc = ", ".join([f"{name}: {cfg['path']}" for name, cfg in models_config.items()])
        self.log_message.emit(f"模型检测初始化完成: {model_desc}")
    
    # ---------- 配置/控制接口 ----------
    def set_model_enabled(self, model_name: str, enable: bool):
        if model_name in self.models:
            self.enabled_models[model_name] = bool(enable)
            # 禁用时清除累积计数，防止后续误触发
            if not enable:
                self.consecutive_danger_frames[model_name] = 0
            self.log_message.emit(f"模型 {model_name} enabled={enable}")

    def update_model_confidence(self, model_name: str, conf: float):
        if model_name in self.models:
            self.models[model_name].conf_threshold = conf
            self.log_message.emit(f"{model_name} conf 更新为 {conf}")

    def update_model_frame_threshold(self, model_name: str, frame_threshold: int):
        if model_name in self.models:
            self.models[model_name].frame_threshold = frame_threshold
            self.log_message.emit(f"{model_name} 连续帧阈值 更新为 {frame_threshold}")
    def update_config(self, enabled_models, model_confidence, model_thresholds=None):
        with QMutexLocker(self._mutex):
            self.enabled_models = enabled_models
            for k, conf in model_confidence.items():
                if k in self.models:
                    self.models[k].conf_threshold = conf
            # 更新模型阈值
            if model_thresholds:
                for k, threshold in model_thresholds.items():
                    if k in self.models:
                        self.models[k].frame_threshold = threshold

        message = f"模型配置已更新: 启用模型={enabled_models} | 置信度={model_confidence}"
        if model_thresholds:
            message += f" | 阈值={model_thresholds}"
        self.log_message.emit(message)


    # ---------- 加载区域 ----------
    def _load_area_for_view(self):
        if 0 <= self.view_index < len(self.xml_paths):
            xml_path = self.xml_paths[self.view_index]
            self.log_message.emit(f"加载区域: {xml_path}")
            self.area_boxes = self.load_area_from_xml(xml_path)
            self.log_message.emit(f"加载到 {len(self.area_boxes)} 个区域")
        else:
            self.area_boxes = []
            self.log_message.emit("没有对应视角，未加载区域")

    # ---------- 主流程 ----------
    def process_frame(self, frame: np.ndarray):
        self.video_buffer.add_frame(frame)
        
        # 每10秒保存一次视频，使用固定文件名覆盖
        now = time.time()
        if now - self.last_save_time >= 10:
            try:
                self.video_buffer.save_buffer_as_video(output_path=self.current_video_path, include_timestamp=True)
                self.log_message.emit(f"定期视频已保存（覆盖）: {self.current_video_path}")
                self.last_save_time = now
            except Exception as e:
                self.log_message.emit(f"定期保存视频失败: {e}")
        
        self._mutex.lock()
        try:
            out = self._process_frame(frame)
            if self.show_ui:
                rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.proc_frame_ready.emit(qimg.copy())
        except Exception as e:
            import traceback
            self.log_message.emit(f"process_frame 错误: {e}")
            self.log_message.emit(traceback.format_exc())
        finally:
            self._mutex.unlock()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self.width, self.height = w, h
        
        # 仅在首次处理帧时加载和缩放区域
        if not hasattr(self, 'area_loaded') or not self.area_loaded:
            # 处理完第一帧后加载area区域
            self.area_boxes, log_message = view2xml(self.view_index, self.width, self.height)
            self.log_message.emit(log_message)
            self.area_loaded = True

        # 1) 仅对启用模型进行推理
        all_results: Dict[str, List[DetectionResult]] = self.run_inference(frame)

        # 2) 处理检测结果（根据每个模型的 trigger_mode 应用不同策略）
        out_frame = self._process_all_results(frame.copy(), all_results)
        return out_frame
    
    def run_inference(self, frame: np.ndarray) -> Dict[str, List[DetectionResult]]:
        """只对 enabled=True 的模型做推理"""
        results = {}
        for name, model_cfg in self.models.items():
            if not self.enabled_models.get(name, False):
                # self.log_message.emit(f"{name} 模型未启用")
                results[name] = []
                continue
            try:
                y = model_cfg.model(frame, conf=model_cfg.conf_threshold, verbose=False)[0]
            except Exception as e:
                self.log_message.emit(f"{name} 模型推理失败: {e}")
                results[name] = []
                continue

            detections = []
            boxes = y.boxes.xyxy.cpu().numpy() if hasattr(y.boxes, "xyxy") else []
            confs = y.boxes.conf.cpu().numpy() if hasattr(y.boxes, "conf") else []
            cls_ids = y.boxes.cls.cpu().numpy() if hasattr(y.boxes, "cls") else []

            # # 调试日志：记录检测到的所有类
            # all_cls_names = [y.names[int(cls_id)] for cls_id in cls_ids]
            # self.log_message.emit(f"{name} 模型检测到的所有类: {all_cls_names}")
            # self.log_message.emit(f"{name} 模型的目标类: {model_cfg.target_classes}")

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                cls_name = y.names[int(cls_id)]
                if cls_name in model_cfg.target_classes:
                    detections.append(DetectionResult(model_cfg.detection_type, cls_name, float(conf), box.tolist()))
            # self.log_message.emit(f"{name} 模型筛选后的检测结果数量: {len(detections)}")
            results[name] = detections
        return results

    # ---------- 结果处理与报警 ----------
    def _process_all_results(self, frame: np.ndarray, all_results: Dict[str, List[DetectionResult]]) -> np.ndarray:
        now = time.time()

        # 若处在报警展示期，判断是否结束并直接绘制现有画面（不更新计数）
        if self.alert_active:
            if now - self.alert_start_time > self.ALERT_DISPLAY_SECONDS:
                self.alert_active = False
                # 报警结束后，重置所有计数
                for k in self.consecutive_danger_frames:
                    self.consecutive_danger_frames[k] = 0
                self.log_message.emit("报警已结束，计数已重置")
            return self._draw_all_detections(frame, all_results, {}, {})

        danger_detected = {k: False for k in self.models.keys()}
        danger_boxes = {k: [] for k in self.models.keys()}

        # 对每个模型应用对应的触发策略
        for name, model_cfg in self.models.items():
            if not self.enabled_models.get(name, False):
                continue
            detections = all_results.get(name, []) or []

            if model_cfg.trigger_mode == 'area':
                # 需要 bbox 完全在某个 area 内才算危险（glove）
                for det in detections:
                    for area in self.area_boxes:
                        if self._box_fully_contains(area, det.bbox):
                            danger_detected[name] = True
                            danger_boxes[name].append(det)
                            break
            else:  # 'any'
                if len(detections) > 0:
                    danger_detected[name] = True
                    danger_boxes[name].extend(detections)

        # 更新计数，只对启用模型计数
        for name in self.models.keys():
            if not self.enabled_models.get(name, False):
                self.consecutive_danger_frames[name] = 0
                continue
            if danger_detected.get(name, False):
                self.consecutive_danger_frames[name] += 1
                # logging on first detection
                if self.consecutive_danger_frames[name] == 1:
                    self.log_message.emit(f"{name} 首次检测到危险（进入计数）")
                # log when reach threshold
                if self.consecutive_danger_frames[name] == self.models[name].frame_threshold:
                    self.log_message.emit(f"{name} 达到阈值: {self.models[name].frame_threshold} 帧")
            else:
                if self.consecutive_danger_frames[name] > 0:
                    self.log_message.emit(f"{name} 危险解除（计数重置）")
                self.consecutive_danger_frames[name] = 0

        # 检查是否达到任一模型的报警条件（只要任一启用模型达阈值就报警）
        should_alert = False
        alert_parts = []
        for name, model_cfg in self.models.items():
            if not self.enabled_models.get(name, False):
                continue
            if self.consecutive_danger_frames.get(name, 0) >= model_cfg.frame_threshold:
                should_alert = True
                # 可展示更友好的消息
                if name == 'glove':
                    alert_parts.append("未佩戴手套")
                elif name == 'head':
                    alert_parts.append("摸头动作")
                else:
                    alert_parts.append(name)

        if should_alert and not self.alert_active:
            # 检查是否需要触发新报警：计算当前报警框与上次报警框的相似度
            need_new_alert = False
            current_alert_boxes = {}
            
            # 收集当前所有报警框
            for name in self.models.keys():
                if not self.enabled_models.get(name, False):
                    continue
                if self.consecutive_danger_frames.get(name, 0) >= self.models[name].frame_threshold:
                    current_alert_boxes[name] = danger_boxes.get(name, [])
            
            # 判断是否需要触发新报警
            if not self.last_alert_boxes:  # 第一次报警，直接触发
                need_new_alert = True
                self.log_message.emit(f"[报警] 首次报警触发，记录报警框信息")
            else:
                self.log_message.emit(f"[报警相似度] 开始计算当前报警与上次报警的相似度")
                # 检查每个模型的报警框
                for model_name, current_boxes in current_alert_boxes.items():
                    last_boxes = self.last_alert_boxes.get(model_name, [])
                    if not last_boxes or not current_boxes:
                        need_new_alert = True
                        self.log_message.emit(f"[报警相似度] 模型: {model_name}, 上次或当前无报警框，触发新报警")
                        break
                    
                    # 计算相似度最高的框
                    max_iou = 0.0
                    for current_box in current_boxes:
                        for last_box in last_boxes:
                            iou = self._calculate_iou(current_box.bbox, last_box.bbox)
                            # 输出前后两次报警框的相似度
                            self.log_message.emit(f"[报警相似度] 模型: {model_name}, 当前框: {current_box.bbox}, 上次框: {last_box.bbox}, 相似度: {iou:.2f}")
                            if iou > max_iou:
                                max_iou = iou
                    
                    # 记录最大相似度
                    self.log_message.emit(f"[报警相似度] 模型: {model_name}, 最高相似度: {max_iou:.2f}")
                    
                    # 如果相似度低于50%，触发新报警
                    if max_iou < 0.5:
                        need_new_alert = True
                        self.log_message.emit(f"[报警相似度] 模型: {model_name}, 相似度低于50%，触发新报警")
                        break
                    else:
                        self.log_message.emit(f"[报警相似度] 模型: {model_name}, 相似度高于50%，不触发新报警")
            
            # 如果需要触发新报警
            if need_new_alert:
                self.alert_active = True
                self.alert_start_time = now
                self.alert_parts = alert_parts  # 保存报警部分信息
                alert_msg = "检测到" + "和".join(alert_parts) + "！"
                self.alert_message.emit(alert_msg)
                self.log_message.emit(f"[报警] {alert_msg}")
                
                # 更新上次报警信息
                self.last_alert_boxes = {}
                for model_name, boxes in current_alert_boxes.items():
                    self.last_alert_boxes[model_name] = boxes.copy()
                self.last_alert_time = now

            # 保存帧与视频：使用唯一文件名避免冲突
            original_frame = frame.copy()
            alert_frame = self._draw_all_detections(frame.copy(), all_results, danger_detected, danger_boxes)
            
            # 只有当需要触发新报警时，才保存图片、发送邮件并重置计数
            if need_new_alert:
                # 保存对比图片到专门文件夹
                try:
                    # 创建报警图片保存文件夹
                    alert_images_dir = os.path.join(os.path.dirname(__file__), '..', 'alert_images')
                    os.makedirs(alert_images_dir, exist_ok=True)
                    
                    # 生成唯一ID
                    unique_id = uuid4().hex
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # 保存原始图片和带标注的报警图片
                    original_path = os.path.join(alert_images_dir, f"alert_{ts}_{unique_id}_original.jpg")
                    annotated_path = os.path.join(alert_images_dir, f"alert_{ts}_{unique_id}_annotated.jpg")
                    
                    cv2.imwrite(original_path, original_frame)
                    cv2.imwrite(annotated_path, alert_frame)
                    
                    self.log_message.emit(f"报警对比图片已保存: {original_path} 和 {annotated_path}")
                except Exception as e:
                    self.log_message.emit(f"保存报警图片失败: {e}")

                # 使用当前保存的视频文件发送邮件
                tmp_video_path = self.current_video_path
                if os.path.exists(tmp_video_path):
                    self.log_message.emit(f"使用当前视频发送报警邮件: {tmp_video_path}")
                else:
                    # 如果当前视频文件不存在，临时保存一个
                    try:
                        unique_id = uuid4().hex
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        tmp_video_path = self.video_buffer.save_buffer_as_video(output_path=os.path.join(self.buffer_dir, f"alert_{ts}_{unique_id}.mp4"),
                                                                           include_timestamp=True)
                        self.log_message.emit(f"临时报警视频已保存: {tmp_video_path}")
                    except Exception as e:
                        self.log_message.emit(f"保存报警视频失败: {e}")
                        tmp_video_path = None

                # 异步发送邮件（线程内删除该唯一文件）
                import threading
                t = threading.Thread(target=self._send_alert_email_thread,
                                     args=(self.video_name, alert_msg, alert_frame, original_frame, tmp_video_path))
                t.daemon = True
                t.start()

                # 报警后立即重置触发模型的计数，防止连续重复报警
                for name in self.models.keys():
                    self.consecutive_danger_frames[name] = 0

        # 最终绘制并返回
        return self._draw_all_detections(frame, all_results, danger_detected, danger_boxes)
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        计算两个检测框的IOU（交并比）
        box1, box2: [x1, y1, x2, y2]
        """
        # 计算交集坐标
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 计算交集面积
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算两个框的面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 计算并集面积
        union_area = box1_area + box2_area - intersection_area
        
        # 计算IOU
        if union_area == 0:
            return 0.0
        return intersection_area / union_area

    def _draw_all_detections(self, frame: np.ndarray, all_results: Dict[str, List[DetectionResult]],
                             danger_detected: Dict[str, bool], danger_boxes: Dict[str, List[DetectionResult]]) -> np.ndarray:
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        now = time.time()
        
        # 直接使用固定字体大小，确保所有视频输出使用完全相同的字号
        # 不再根据视频分辨率动态调整，避免不同视频字号不一致的问题
        # 将字体大小设置为更大的值，确保清晰可见
        font_scale = 1.8  # 统一的大字体大小，适用于所有视频输出
        
        # 计算文字位置偏移和线条粗细，保持比例
        line_thickness = max(2, int(font_scale * 2.5))  # 增加线条粗细
        text_offset = int(40 * font_scale / 0.8)  # 基于默认0.8字体大小的比例调整，增加垂直偏移
        text_spacing = int(40 * font_scale / 0.8)  # 基于默认0.8字体大小的比例调整，增加间距

        # 报警提示和倒计时
        if self.alert_active:
            # 计算剩余时间
            remaining_time = max(0, self.ALERT_DISPLAY_SECONDS - (now - self.alert_start_time))
            countdown_text = f"Alarm: {remaining_time:.1f}s"
            
            # 绘制报警消息 - 使用英文避免中文显示问题
            alert_msg = "ALERT: Abnormal behavior detected!"
            if hasattr(self, 'alert_parts') and self.alert_parts:
                # 将中文报警类型转换为英文
                english_parts = []
                for part in self.alert_parts:
                    if part == "未佩戴手套":
                        english_parts.append("No glove")
                    elif part == "摸头动作":
                        english_parts.append("Head touching")
                    else:
                        english_parts.append(part)
                alert_msg = "ALERT: " + " and ".join(english_parts) + " detected!"
            
            # 使用默认字体，避免中文显示问题
            cv2.putText(annotated, alert_msg, (20, text_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness)
            cv2.putText(annotated, countdown_text, (20, text_offset + text_spacing), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_thickness)
        
        # area 区域绘制（仅当有区域且 glove 启用时绘制）
        if self.area_boxes and self.enabled_models.get('glove', False):
            for box in self.area_boxes:
                x1, y1, x2, y2 = map(int, box)
                box_thickness = max(1, int(font_scale * 2))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), box_thickness)
                cv2.putText(annotated, "area", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_thickness)

        # 绘制每个启用模型的检测结果
        for name, detections in all_results.items():
            if not self.enabled_models.get(name, False):
                continue
            # 设置普通检测状态的颜色：glove用蓝色，touch用绿色（不再用红色，避免与danger状态混淆）
            color = (255, 0, 0) if name == 'glove' else (0, 255, 0)
            label_prefix = "bare" if name == 'glove' else "touch"
            for det in detections:
                x1, y1, x2, y2 = map(int, det.bbox)
                is_danger = any(d.bbox_equals(det.bbox) for d in danger_boxes.get(name, []))
                
                if is_danger:
                    # 增强的 danger 框效果
                    box_color = (0, 0, 255)  # 红色
                    box_thickness = max(2, int(font_scale * 3))  # 更粗的边框
                    # 绘制双重边框
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), box_thickness)
                    cv2.rectangle(annotated, (x1+2, y1+2), (x2-2, y2-2), (0, 0, 255), box_thickness-1)
                    # 危险标签
                    label = f"DANGER: {label_prefix} ({det.confidence:.2f})"
                    label_color = (0, 0, 255)
                else:
                    # 普通检测框
                    box_color = color
                    box_thickness = max(1, int(font_scale * 2))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thickness)
                    label = f"{label_prefix}: {det.confidence:.2f}"
                    label_color = box_color
                
                # 绘制标签
                cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, line_thickness)

        return annotated

    def _box_fully_contains(self, container_box: List[float], inner_box: List[float]) -> bool:
        x1, y1, x2, y2 = [float(x) for x in container_box]
        x3, y3, x4, y4 = [float(x) for x in inner_box]
        return (x3 >= x1) and (y3 >= y1) and (x4 <= x2) and (y4 <= y2)

    # ---------- 邮件线程 ----------
    def _send_alert_email_thread(self, video_name, alert_message, alert_frame, original_frame, video_path):
        try:
            # 这里 email_sender 的签名可能与现有不同，按你的实现调整参数顺序
            self.email_sender.send_alert_email(video_name, alert_message, alert_frame, self.alert_email, original_frame, video_path)
        except Exception as e:
            self.log_message.emit(f"发送报警邮件失败: {e}")
        finally:
            # 只删除临时生成的视频文件，不删除定期保存的当前视频
            if video_path and os.path.exists(video_path) and not video_path == self.current_video_path:
                try:
                    os.remove(video_path)
                    self.log_message.emit(f"已删除临时视频: {video_path}")
                except Exception as e:
                    self.log_message.emit(f"删除临时视频失败: {e}")

    # ---------- XML 加载（与原逻辑类似） ----------
    def load_area_from_xml(self, xml_path: str) -> List[List[float]]:
        area_boxes = []
        if not os.path.exists(xml_path):
            self.log_message.emit(f"XML不存在: {xml_path}")
            return area_boxes
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size_node = root.find("size")
            xml_w = int(size_node.find("width").text) if size_node is not None and size_node.find("width") is not None else None
            xml_h = int(size_node.find("height").text) if size_node is not None and size_node.find("height") is not None else None
            raw = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name == 'area':
                    bnd = obj.find('bndbox')
                    xmin = int(float(bnd.find('xmin').text))
                    ymin = int(float(bnd.find('ymin').text))
                    xmax = int(float(bnd.find('xmax').text))
                    ymax = int(float(bnd.find('ymax').text))
                    raw.append([xmin, ymin, xmax, ymax])
            # 缩放
            if self.width is None or self.height is None:
                return [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in raw]
            tw, th = int(self.width), int(self.height)
            if xml_w and xml_h:
                sx = tw / xml_w
                sy = th / xml_h
                for x1, y1, x2, y2 in raw:
                    nx1 = max(0, min(tw - 1, int(round(x1 * sx))))
                    ny1 = max(0, min(th - 1, int(round(y1 * sy))))
                    nx2 = max(0, min(tw - 1, int(round(x2 * sx))))
                    ny2 = max(0, min(th - 1, int(round(y2 * sy))))
                    area_boxes.append([nx1, ny1, nx2, ny2])
            return area_boxes
        except Exception as e:
            self.log_message.emit(f"解析 XML 失败: {e}")
            return []