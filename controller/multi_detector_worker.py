"""
多模型检测工作类 - 支持同时运行多个YOLO模型进行检测
"""

import time
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QMutex
from PyQt6.QtGui import QImage
from ultralytics import YOLO

from model.email_sender import EmailSender
from model.video_buffer_manager import VideoBufferManager


class DetectionResult:
    """单个检测结果"""
    def __init__(self, detection_type: str, class_name: str, confidence: float, bbox: List[float]):
        self.detection_type = detection_type  # 'glove' 或 'head'
        self.class_name = class_name  # 'bare' 或 'touch'
        self.confidence = confidence
        self.bbox = bbox  # [x1, y1, x2, y2]


class DetectionModel:
    """检测模型配置"""
    def __init__(self, model_path: str, detection_type: str, target_classes: List[str], conf_threshold: float):
        self.model = YOLO(model_path)
        device = 'cuda'
        self.model.to(device)
        self.detection_type = detection_type  # 'glove' 或 'head'
        self.target_classes = target_classes  # 需要检测的类别列表
        self.conf_threshold = conf_threshold  # 置信度阈值


class MultiDetectorWorker(QObject):
    """多模型检测工作类"""
    proc_frame_ready = pyqtSignal(QImage)  # 处理后的帧信号
    log_message = pyqtSignal(str)  # 日志信号
    alert_message = pyqtSignal(str)  # 报警信号

    # 报警参数配置
    ALERT_FRAME_THRESHOLD = 5  # 连续多少帧危险才触发报警
    ALERT_DISPLAY_SECONDS = 5  # 报警持续显示时间（秒）

    def __init__(self, glove_model_path: str, head_model_path: str, video_name: str, 
                 view_index: int, alert_email: str, parent=None):
        super().__init__(parent)
        
        # 初始化多个检测模型
        self.models = {
            'glove': DetectionModel(glove_model_path, 'glove', ['bare'], 0.8),
            'head': DetectionModel(head_model_path, 'head', ['touch'], 0.5)
        }
        
        self.video_name = video_name
        self.alert_email = alert_email
        self.view_index = view_index
        
        # 线程安全锁
        self._mutex = QMutex()
        
        # 报警控制变量 - 分别跟踪不同类型的检测
        self.consecutive_danger_frames = {
            'glove': 0,
            'head': 0
        }
        self.alert_active = False
        self.alert_start_time = 0
        self.show_ui = True
        
        # 区域检测相关变量（仅手套检测使用）
        self.area_boxes = []
        self.view_names = ["视角1", "视角2"]
        self.xml_paths = [
            os.path.join(os.path.dirname(__file__), "..", "area", "0911_1_frame00000.xml"), 
            os.path.join(os.path.dirname(__file__), "..", "area", "0911_2_frame00000.xml") 
        ]
        
        # 邮件发送器
        self.email_sender = EmailSender()
        self.processed_alert_frame = None
        
        # 视频缓冲管理器，用于存储最近30秒的视频帧
        self.video_buffer = VideoBufferManager(buffer_seconds=30, fps=30)
        
        # 加载检测区域（XML文件）- 仅手套检测需要
        # 根据视频名称的对应关系决定是否加载区域
        # 使用传入的view_index判断是否有对应关系
        if self.view_index < len(self.xml_paths) and os.path.exists(self.xml_paths[self.view_index]):
            self.area_boxes = self.load_area_from_xml(self.xml_paths[self.view_index])
            self.log_message.emit(f"检测模式：区域检测（视角{self.view_index + 1}）")
        else:
            # 没有对应关系或文件不存在，使用全画面检测
            self.log_message.emit("检测模式：全画面检测（未找到对应区域文件）")
        
        self.log_message.emit(f"多模型检测器初始化完成 - 手套模型: {glove_model_path}, 头部模型: {head_model_path}")

    def process_frame(self, frame: np.ndarray):
        """主入口：处理帧（线程安全）"""
        # 将当前帧添加到视频缓冲区
        self.video_buffer.add_frame(frame)
        
        self._mutex.lock()
        try:
            annotated_frame = self._process_frame(frame)
            # 转换并发送处理后的帧
            if self.show_ui:
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                q_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.proc_frame_ready.emit(q_img.copy())  # 发送副本避免线程冲突
        except Exception as e:
            self.log_message.emit(f"帧处理错误: {str(e)}")
            import traceback
            self.log_message.emit(f"帧处理详细错误: {traceback.format_exc()}")
        finally:
            self._mutex.unlock()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """核心检测逻辑 - 同时运行所有模型"""
        # 获取视频帧尺寸
        h, w = frame.shape[:2]
        
        # 检查是否首次处理帧或尺寸发生变化
        if not hasattr(self, 'width') or not hasattr(self, 'height') or self.width != w or self.height != h:
            self.width = w
            self.height = h
            self.log_message.emit(f"视频尺寸更新: {w}x{h}")
        
        # 运行所有检测模型
        all_results = {}
        annotated_frame = frame.copy()
        
        for model_name, model_config in self.models.items():
            results = self._run_single_model(model_config, frame)
            all_results[model_name] = results
            
        # 处理所有检测结果
        final_frame = self._process_all_results(annotated_frame, all_results)
        
        return final_frame

    def _run_single_model(self, model_config: DetectionModel, frame: np.ndarray) -> List[DetectionResult]:
        """运行单个模型检测"""
        results = model_config.model(frame, conf=model_config.conf_threshold, verbose=False)[0]
        detections = []
        
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls):
            cls_name = results.names[int(cls)]
            conf = float(box.conf) if hasattr(box, 'conf') else 0.9
            
            if cls_name in model_config.target_classes:
                detection = DetectionResult(
                    detection_type=model_config.detection_type,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=box.tolist()
                )
                detections.append(detection)
                
        return detections

    def _process_all_results(self, frame: np.ndarray, all_results: Dict[str, List[DetectionResult]]) -> np.ndarray:
        """处理所有模型的检测结果"""
        current_time = time.time()
        
        # 情况A：如果正在报警中，检查是否应该结束
        if self.alert_active:
            if current_time - self.alert_start_time > self.ALERT_DISPLAY_SECONDS:
                self.alert_active = False
                # 重置所有计数器
                for key in self.consecutive_danger_frames:
                    self.consecutive_danger_frames[key] = 0
                self.log_message.emit("报警状态已重置")
            # 直接返回当前帧（报警期间不处理新检测）
            return self._draw_all_detections(frame, all_results, {}, {})
        
        # 分别处理每种检测结果
        danger_detected = {
            'glove': False,
            'head': False
        }
        
        danger_boxes = {
            'glove': [],
            'head': []
        }
        
        # 处理手套检测结果
        if 'glove' in all_results:
            glove_results = all_results['glove']
            glove_danger_boxes = []
            
            for detection in glove_results:
                # 检查是否进入危险区域
                for area_idx, area_box in enumerate(self.area_boxes):
                    if self._box_fully_contains(area_box, detection.bbox):
                        danger_detected['glove'] = True
                        glove_danger_boxes.append(detection)
                        break
                        
            danger_boxes['glove'] = glove_danger_boxes
        
        # 处理头部检测结果
        if 'head' in all_results:
            head_results = all_results['head']
            head_danger_boxes = []
            
            for detection in head_results:
                danger_detected['head'] = True
                head_danger_boxes.append(detection)
                
            danger_boxes['head'] = head_danger_boxes
        
        # 更新连续危险帧计数
        for detection_type in danger_detected:
            if danger_detected[detection_type]:
                self.consecutive_danger_frames[detection_type] += 1
                # 只在首次检测到时输出
                if self.consecutive_danger_frames[detection_type] == 1:
                    if detection_type == 'glove':
                        self.log_message.emit("检测到未佩戴手套")
                    elif detection_type == 'head':
                        self.log_message.emit("检测到摸头动作")
                elif self.consecutive_danger_frames[detection_type] == self.ALERT_FRAME_THRESHOLD:
                    self.log_message.emit(f"{detection_type}检测连续危险帧达到阈值")
            else:
                if self.consecutive_danger_frames[detection_type] > 0:
                    self.log_message.emit(f"{detection_type}检测危险状态解除")
                self.consecutive_danger_frames[detection_type] = 0
        
        # 检查是否满足报警条件（任一类型达到阈值）
        should_alert = any(count >= self.ALERT_FRAME_THRESHOLD for count in self.consecutive_danger_frames.values())
        
        if should_alert and not self.alert_active:
            self.alert_active = True
            self.alert_start_time = current_time
            
            # 构建报警消息
            alert_parts = []
            if self.consecutive_danger_frames['glove'] >= self.ALERT_FRAME_THRESHOLD:
                alert_parts.append("未佩戴手套")
            if self.consecutive_danger_frames['head'] >= self.ALERT_FRAME_THRESHOLD:
                alert_parts.append("摸头动作")
                
            alert_msg = "检测到" + "和".join(alert_parts) + "！"
            self.alert_message.emit(alert_msg)
            self.log_message.emit(f"[报警] {alert_msg}")
            
            # 保存原始帧（无标注）和绘制了检测框的帧
            original_frame = frame.copy()  # 保存原始无标注帧
            alert_frame = self._draw_all_detections(frame, all_results, danger_detected, danger_boxes)
            self._send_alert_email(alert_msg, alert_frame, original_frame)
            
            # 重置计数器
            for key in self.consecutive_danger_frames:
                self.consecutive_danger_frames[key] = 0
        
        return self._draw_all_detections(frame, all_results, danger_detected, danger_boxes)

    def _draw_all_detections(self, frame: np.ndarray, all_results: Dict[str, List[DetectionResult]], 
                           danger_detected: Dict[str, bool], danger_boxes: Dict[str, List[DetectionResult]]) -> np.ndarray:
        """绘制所有检测结果"""
        annotated_frame = frame.copy()
        
        # 绘制区域框（仅手套检测）
        if self.area_boxes:
            for box in self.area_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated_frame, "area", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制所有检测结果
        for model_name, results in all_results.items():
            color = (255, 0, 0) if model_name == 'glove' else (0, 0, 255)  # 蓝色手套，红色头部
            label_prefix = "bare" if model_name == 'glove' else "touch"
            
            for detection in results:
                x1, y1, x2, y2 = map(int, detection.bbox)
                
                # 危险框用红色，普通框用模型颜色
                if detection in danger_boxes.get(model_name, []):
                    box_color = (0, 0, 255)  # 红色表示危险
                    thickness = 3
                else:
                    box_color = color
                    thickness = 2
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness)
                
                # 添加标签
                label = f"{label_prefix}: {detection.confidence:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        return annotated_frame

    def _box_fully_contains(self, container_box: List[float], inner_box: List[float]) -> bool:
        """检查内部框是否完全包含在容器框内"""
        x1, y1, x2, y2 = container_box
        x3, y3, x4, y4 = inner_box
        return (x3 >= x1) and (y3 >= y1) and (x4 <= x2) and (y4 <= y2)

    def _send_alert_email(self, alert_message: str, frame: np.ndarray, original_frame: np.ndarray = None):
        """发送报警邮件，包含视频缓冲"""
        if self.alert_email and frame is not None:
            import threading
            
            # 保存视频缓冲为文件（在报警时）
            video_path = None
            try:
                # 将缓冲区中的帧保存为视频文件
                video_path = self.video_buffer.save_buffer_as_video(include_timestamp=True)
                print(f"报警视频已保存: {video_path}")
            except Exception as e:
                print(f"保存报警视频时出错: {str(e)}")
            
            thread = threading.Thread(
                target=self._send_alert_email_thread,
                args=(self.video_name, alert_message, frame, original_frame, video_path)
            )
            thread.daemon = True
            thread.start()
    
    def _send_alert_email_thread(self, video_name, alert_message, frame, original_frame, video_path):
        """邮件发送线程函数，支持发送视频附件"""
        try:
            # 调用邮件发送器发送报警邮件
            self.email_sender.send_alert_email(
                video_name, 
                alert_message, 
                frame, 
                self.alert_email, 
                original_frame, 
                video_path
            )
        finally:
            # 邮件发送完成后，清理临时视频文件
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    print(f"已清理临时视频文件: {video_path}")
                except Exception as e:
                    print(f"清理临时视频文件时出错: {str(e)}")

    def load_area_from_xml(self, xml_path: str) -> List[List[float]]:
        """从XML文件加载检测区域"""
        area_boxes = []
        if not os.path.exists(xml_path):
            self.log_message.emit(f"XML文件不存在: {xml_path}")
            return area_boxes
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size_node = root.find("size")
            xml_w = int(size_node.find("width").text) if size_node is not None and size_node.find("width") is not None else None
            xml_h = int(size_node.find("height").text) if size_node is not None and size_node.find("height") is not None else None
            
            raw_boxes = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name == 'area':  # 只加载类别为area的区域
                    bndbox = obj.find('bndbox')
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    raw_boxes.append([xmin, ymin, xmax, ymax])
            
            # 缩放区域到当前视频尺寸
            if not hasattr(self, "width") or not hasattr(self, "height"):
                return [[int(xmin), int(ymin), int(xmax), int(ymax)] for xmin, ymin, xmax, ymax in raw_boxes]
            
            tw, th = int(self.width), int(self.height)
            if xml_w and xml_h:
                sx = tw / xml_w
                sy = th / xml_h
                for xmin, ymin, xmax, ymax in raw_boxes:
                    nx1 = max(0, min(tw - 1, int(round(xmin * sx))))
                    ny1 = max(0, min(th - 1, int(round(ymin * sy))))
                    nx2 = max(0, min(tw - 1, int(round(xmax * sx))))
                    ny2 = max(0, min(th - 1, int(round(ymax * sy))))
                    area_boxes.append([nx1, ny1, nx2, ny2])
            
            self.log_message.emit(f"已加载 {len(area_boxes)} 个检测区域")
            return area_boxes
            
        except Exception as e:
            self.log_message.emit(f"加载XML文件 {xml_path} 时出错: {e}")
            return []