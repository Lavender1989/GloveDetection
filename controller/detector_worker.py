"""
报警逻辑修改位置
"""

import math
import time
import os
import xml.etree.ElementTree as ET
from collections import deque

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QMutex
from PyQt6.QtGui import QImage
from ultralytics import YOLO

from model.email_sender import EmailSender


class DetectorWorker(QObject):
    proc_frame_ready = pyqtSignal(QImage)  # 处理后的帧信号
    log_message = pyqtSignal(str)  # 日志信号
    alert_message = pyqtSignal(str)  # 报警信号

    # 报警参数配置
    ALERT_FRAME_THRESHOLD = 10  # 连续多少帧危险才触发报警
    ALERT_DISPLAY_SECONDS = 5  # 报警持续显示时间（秒）

    # 修改 __init__ 方法
    def __init__(self, model_path, video_name, view_index, alert_email, parent=None):
        super().__init__(parent)
        self.model = YOLO(model_path)
        device = 'cuda'
        self.model.to(device)
        self.log_message.emit(f"模型运行设备: {device}")
        
        # 线程安全锁
        self._mutex = QMutex()
    
        # 报警控制变量
        self.consecutive_danger_frames = 0
        self.alert_active = False  # 当前是否在报警中
        self.alert_start_time = 0  # 报警开始时间戳
        self.show_ui = True
    
        # 区域检测相关变量
        self.area_boxes = []
        self.current_view = view_index  # 直接使用传入的视角索引
        self.view_names = ["视角1", "视角2"]
        # 确保XML路径正确
        self.xml_paths = [
            os.path.join(os.path.dirname(__file__), "..", "area", "0911_1_frame00000.xml"), 
            os.path.join(os.path.dirname(__file__), "..", "area", "0911_2_frame00000.xml") 
        ]
    
        # 视频名称和报警邮箱
        self.video_name = video_name
        self.alert_email = alert_email
        self.email_sender = EmailSender()  # 邮件发送器实例
    
        # 保存报警帧
        self.processed_alert_frame = None
    
        # 直接加载区域配置
        # self.area_boxes = self.load_area_from_xml(self.xml_paths[self.current_view])
        # self.log_message.emit(f"已加载 {self.view_names[self.current_view]} 对应的区域配置")

    def process_frame(self, frame: np.ndarray):
        """主入口：处理帧（线程安全）"""

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

    def _process_frame(self, frame):
        """核心检测逻辑"""
        # 获取视频帧尺寸
        h, w = frame.shape[:2]
        # 检查是否首次处理帧或尺寸发生变化
        if not hasattr(self, 'width') or not hasattr(self, 'height') or self.width != w or self.height != h:
            # 设置尺寸属性
            self.width = w
            self.height = h
            # 重新加载并缩放检测区域
            self.area_boxes = self.load_area_from_xml(self.xml_paths[self.current_view])
            self.log_message.emit(f"已根据视频尺寸 {w}x{h} 重新加载并缩放检测区域")
        
        # 1. 模型推理
        results = self.model(frame, conf=0.8, verbose=False)[0]
        annotated_frame = frame.copy()
        danger_detected = False
        bare_boxes = []
        danger_boxes = []
        detected_area_indices = []  # 记录检测到危险的区域索引

        # 2. 提取检测框
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls):
            cls_name = results.names[int(cls)]
            if cls_name == 'bare':
                bare_boxes.append(box)

        # 3. 报警状态检查
        current_time = time.time()

        # 情况A：如果正在报警中，检查是否应该结束
        if self.alert_active:
            if current_time - self.alert_start_time > self.ALERT_DISPLAY_SECONDS:
                self.alert_active = False
                self.consecutive_danger_frames = 0
                self.log_message.emit(f"报警状态已重置")
            # 直接返回当前帧（报警期间不处理新检测）
            return self._draw_detections(annotated_frame, bare_boxes, danger_boxes)

        # 情况B：正常检测危险
        # 3.1 判断是否有危险（未戴手套的手完全进入危险区域）
        for bare_box_idx, bare_box in enumerate(bare_boxes):
            for area_idx, area_box in enumerate(self.area_boxes):
                if self.box_fully_contains(area_box, bare_box):
                    danger_detected = True
                    danger_boxes.append(bare_box)
                    detected_area_indices.append(area_idx)
                    break

        # 3.2 记录危险状态
        if danger_detected:
            self.consecutive_danger_frames += 1
            unique_areas = set(detected_area_indices)
            area_info = f"区域{', '.join(map(str, unique_areas))}"
            # self.log_message.emit(f"检测到未佩戴手套 (区域: {area_info}, 连续危险帧: {self.consecutive_danger_frames})")
        else:
            if self.consecutive_danger_frames > 0:
                pass
                # self.log_message.emit(f"未检测到危险，重置连续危险帧计数 (之前计数: {self.consecutive_danger_frames})")
            self.consecutive_danger_frames = 0

        # 3.3 检查是否满足报警条件
        if self.consecutive_danger_frames >= self.ALERT_FRAME_THRESHOLD and not self.alert_active:
            self.alert_active = True
            self.alert_start_time = current_time
            alert_msg = "检测到未佩戴手套操作！"
            self.alert_message.emit(alert_msg)
            self.log_message.emit(f"[报警] {alert_msg}")
            # 绘制检测结果
            self.processed_alert_frame = self._draw_detections(annotated_frame, bare_boxes, danger_boxes)
            # 发送报警邮件
            self.send_alert_email(alert_msg)
            self.consecutive_danger_frames = 0  # 触发报警后重置计数

        return self._draw_detections(annotated_frame, bare_boxes, danger_boxes)

    # ---------------------- 辅助方法 ----------------------
    def send_alert_email(self, alert_message):
        """发送报警邮件（使用处理后的帧）"""
        if self.alert_email and self.processed_alert_frame is not None:
            # 在后台线程中发送邮件，避免阻塞主线程
            import threading
            thread = threading.Thread(
                target=self.email_sender.send_alert_email,
                args=(self.video_name, alert_message, self.processed_alert_frame, self.alert_email)
            )
            thread.daemon = True
            thread.start()

    def load_area_from_xml(self, xml_path):
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
            self.log_message.emit(f"成功从 {xml_path} 加载了 {len(raw_boxes)} 个检测区域")
            if not hasattr(self, "width") or not hasattr(self, "height"):
                print("警告: self.width/self.height 未设置，返回原始坐标（未缩放）")
                return [[int(xmin), int(ymin), int(xmax), int(ymax)] for xmin, ymin, xmax, ymax in raw_boxes]
            tw, th = int(self.width), int(self.height)
            scaled = []
            if xml_w and xml_h:
                sx = tw / xml_w
                sy = th / xml_h
                for xmin, ymin, xmax, ymax in raw_boxes:
                    nx1 = int(round(xmin * sx))
                    ny1 = int(round(ymin * sy))
                    nx2 = int(round(xmax * sx))
                    ny2 = int(round(ymax * sy))
                    # 裁剪到图像范围
                    nx1 = max(0, min(tw - 1, nx1))
                    nx2 = max(0, min(tw - 1, nx2))
                    ny1 = max(0, min(th - 1, ny1))
                    ny2 = max(0, min(th - 1, ny2))
                    scaled.append([nx1, ny1, nx2, ny2])
                self.log_message.emit(f"[XML缩放] {xml_w}x{xml_h} -> {tw}x{th}, boxes={len(scaled)}")
                return scaled
        except Exception as e:
            self.log_message.emit(f"加载XML文件 {xml_path} 时出错: {e}")
            return []

    def box_fully_contains(self, box1, box2):
        """检查box2是否完全包含在box1内"""
        x1, y1, x2, y2 = box1  # 区域框
        x3, y3, x4, y4 = box2  # bare框
        result = (x3 >= x1) and (y3 >= y1) and (x4 <= x2) and (y4 <= y2)
        # self.log_message.emit(f"检查区域框 ({x1}, {y1}, {x2}, {y2}) 是否完全包含bare框 ({x3}, {y3}, {x4}, {y4}): {result}")
        return result

    def get_view_by_video_name(self, video_path):
        """根据视频文件名或RTSP地址确定使用哪个视角"""
        from .video_view_mapping import get_view_for_video, get_view_name
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        video_name = os.path.basename(video_path)
        
        # 使用新的映射文件获取视角
        view_index = get_view_for_video(video_path)
        view_name = get_view_name(view_index)
        
        # 输出日志
        self.log_message.emit(f"[{current_time}] {video_name}: 成功加载{view_name}")
        return view_index

    def _draw_detections(self, frame, bare_boxes, danger_boxes):
        """绘制检测结果"""
        # 绘制区域框
        for box in self.area_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "area", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 绘制未戴手套框（蓝色）
        for box in bare_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "bare", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 绘制危险框（红色）
        for box in danger_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "DANGER", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示当前状态和视角
        status_text = "DANGER" if self.alert_active else "SAFE"
        status_color = (0, 0, 255) if self.alert_active else (0, 255, 0)
        cv2.putText(frame, f"status: {status_text}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

        if self.alert_active:
            current_time = time.time()
            alert_duration = current_time - self.alert_start_time
            cv2.putText(frame, "ALARM! BARE HANDS DETECTED!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Duration: {int(alert_duration)}s", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame