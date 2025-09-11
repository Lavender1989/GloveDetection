"""
报警逻辑修改位置
"""


import math
import time
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
    ALERT_IOU_THRESH = 0.01  # 交并比阈值
    ALERT_DISTANCE_THRESH = 150  # 中心距离阈值（像素）
    ALERT_FRAME_THRESHOLD = 10  # 连续多少帧危险才触发报警
    ALERT_DISPLAY_SECONDS = 5  # 报警持续显示时间（秒）


    def __init__(self, model_path,video_name, alert_email, parent=None):
        super().__init__(parent)
        self.model = YOLO(model_path)
        device = 'cuda' if self.model.device.type == 'cuda' else 'cpu'
        self.model.to(device)
        self.log_message.emit(f"模型运行设备: {device}")
        
        # 添加下面的代码以显示设备信息对话框
        # from PyQt6.QtWidgets import QMessageBox
        # QMessageBox.information(None, "设备信息", f"模型运行设备: {device}")
        
        # 线程安全锁
        self._mutex = QMutex()

        # 报警控制变量
        self.recent_violations = deque(maxlen=self.ALERT_FRAME_THRESHOLD)  # 危险帧记录队列
        self.alert_active = False  # 当前是否在报警中
        self.alert_start_time = 0  # 报警开始时间戳
        self.show_ui = True

        # 新增：视频名称和报警邮箱
        self.video_name = video_name
        self.alert_email = alert_email
        self.email_sender = EmailSender()  # 邮件发送器实例

        # 新增：保存报警帧
        self.processed_alert_frame = None

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
        """核心检测逻辑（修改重点）"""
        # 1. 模型推理
        results = self.model(frame, conf=0.5, verbose=False)[0]
        annotated_frame = frame.copy()
        danger_detected = False
        car_boxes, bare_boxes = [], []

        # 2. 提取检测框
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls):
            cls_name = results.names[int(cls)]
            if cls_name == 'car':
                car_boxes.append(box)
            elif cls_name == 'bare':
                bare_boxes.append(box)

        # 3. 报警状态检查（关键修改点）
        current_time = time.time()

        # 情况A：如果正在报警中，检查是否应该结束
        if self.alert_active:
            if current_time - self.alert_start_time > self.ALERT_DISPLAY_SECONDS:
                self.alert_active = False
                self.recent_violations.clear()
                self.log_message.emit(f"报警状态已重置")
            # 直接返回当前帧（报警期间不处理新检测）
            return self._draw_detections(annotated_frame, car_boxes, bare_boxes, [])

        # 情况B：正常检测危险
        # 3.1 判断是否有危险（汽车和未戴手套的手距离过近）
        danger_pairs = []
        for bare_box in bare_boxes:
            for car_box in car_boxes:
                iou = self._box_iou(bare_box, car_box)
                dist = self._center_distance(bare_box, car_box)
                if iou > self.ALERT_IOU_THRESH or dist < self.ALERT_DISTANCE_THRESH:
                    danger_detected = True
                    danger_pairs.append((car_box, bare_box))
                    break  # 只要有一个危险对就停止检测

        # 3.2 记录危险状态（仅在非报警状态下）
        self.recent_violations.append(danger_detected)

        # 3.3 检查是否满足报警条件（连续多帧危险）
        if sum(self.recent_violations) >= self.ALERT_FRAME_THRESHOLD:
            self.alert_active = True
            self.alert_start_time = current_time
            alert_msg = "检测到未佩戴手套操作！"
            self.alert_message.emit(alert_msg)
            self.log_message.emit(f"[报警] {alert_msg}")
            # 4. 绘制检测结果
            self.processed_alert_frame=self._draw_detections(annotated_frame, car_boxes, bare_boxes, danger_pairs)
            # 新增：发送报警邮件（使用处理后的帧）
            self.send_alert_email(alert_msg)
            self.recent_violations.clear()  # 触发报警后清空记录

        return self._draw_detections(annotated_frame, car_boxes, bare_boxes, danger_pairs)

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

    def _box_iou(self, box1, box2):
        """计算两个框的交并比"""
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return interArea / (box1Area + box2Area - interArea + 1e-6)

    def _box_center(self, box):
        """计算框中心坐标"""
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    def _center_distance(self, box1, box2):
        """计算两个框中心的距离"""
        cx1, cy1 = self._box_center(box1)
        cx2, cy2 = self._box_center(box2)
        return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def _draw_detections(self, frame, car_boxes, bare_boxes, danger_pairs):
        """绘制检测结果"""
        # 绘制汽车框（绿色）
        for box in car_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "car", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制未戴手套框（蓝色）
        for box in bare_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "bare", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 绘制危险区域（红色）
        for car_box, bare_box in danger_pairs:
            x_coords = [car_box[0], car_box[2], bare_box[0], bare_box[2]]
            y_coords = [car_box[1], car_box[3], bare_box[1], bare_box[3]]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
            cv2.putText(frame, "DANGER", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示当前状态
        status_text = "DANGER" if self.alert_active else "SAFE"
        status_color = (0, 0, 255) if self.alert_active else (0, 255, 0)
        cv2.putText(frame, f"status: {status_text}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        return frame
