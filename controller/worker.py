# 主调度类
# MultiDetectorWorker（主调度类）——组装各子模块并暴露 PyQt 信号接口
# 此文件保留原来的信号接口（proc_frame_ready, log_message, alert_message），功能不变
from PyQt6.QtCore import QObject, pyqtSignal, Qt, pyqtSlot
from PyQt6.QtGui import QImage
import queue
import time
import cv2
from typing import Dict
from .types import DetectionModel
from .inference_thread import InferenceThread
from .region_loader import load_area_for_view
from .scheduler import Scheduler
from .postprocess import PostProcessor
from .drawer import Drawer

class MultiDetectorWorker(QObject):
    proc_frame_ready = pyqtSignal(QImage)
    log_message = pyqtSignal(str)
    alert_message = pyqtSignal(str)
    detection_result = pyqtSignal(str, list)  # 发送检测类型和结果列表

    def __init__(self, models_config: Dict[str, Dict], 
    video_name: str, 
    view_index: int, 
    alert_email: str, 
    capture_manager:QObject,
    video_id: str):
        """
        models_config: 模型配置字典
        capture_manager: VideoCaptureManager 实例（外部创建）
        """
        super().__init__()

        # 组装 DetectionModel 对象
        self.models: Dict[str, DetectionModel] = {}
        for name, cfg in models_config.items():
            self.models[name] = DetectionModel(
                name=name,
                model_path=cfg['path'],
                target_classes=cfg.get('target_classes', []),
                # 优先使用models_config中的'conf'参数，其次是'conf_threshold'，最后是默认值0.5（与types.py保持一致）
                conf_threshold=cfg.get('conf', cfg.get('conf_threshold', 0.5)),
                frame_threshold=cfg.get('frame_threshold', 2),
                trigger_mode=cfg.get('trigger_mode', 'area'),
                enabled=cfg.get('enabled', True),
            )

        self.video_name = video_name
        self.view_index = view_index
        self.alert_email = alert_email
        self.capture_manager = capture_manager
        self.video_id = video_id
        # 存储当前邮箱信息，用于检测变化
        self._current_email = alert_email

        # --- tools ---
        self.postproc = PostProcessor(self.models, area_boxes=[], alert_email=self.alert_email)
        self.drawer = Drawer()

        # InferenceThread：单 worker 负责 GPU 推理
        self.inference_thread = InferenceThread(self.models, parent=self)
        self.inference_thread.inference_done.connect(self.on_inference_done, Qt.ConnectionType.QueuedConnection)
        self.inference_thread.inference_error.connect(self.log_message.emit)
        self.inference_thread.start()

        self.scheduler = Scheduler(target_fps=30.0, parent=self)
        self.scheduler.tick.connect(self.on_tick)

        # 状态控制
        self._running = True
        self._main_thread = None
        self._paused = False  # 暂停检测
        self._stopped = False  # 终止检测
        # 区域加载标志位
        self._area_loaded = False
        self._view_not_found_logged = False
        # 邮箱信息显示标志位
        self._email_info_logged = False

        self.log_message.emit(f"MultiDetectorWorker 初始化完成，模型: {list(self.models.keys())}")

    # ----- 兼容原接口：外部依旧可用 process_frame 主动喂帧 -----
    def process_frame(self, frame):
        """兼容外部主动推帧接口：把帧放入 frame_queue（非阻塞）"""
        try:
            self.frame_queue.put_nowait((int(time.time()*1000), frame))
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((int(time.time()*1000), frame))
            except Exception:
                pass

    # ----- inference_thread 的回调（在主线程） -----
    @pyqtSlot(object, object)
    def on_inference_done(self, frame_np, raw_results):
        """
        inference_thread 发回 raw_results（ultralytics 对象）
        在此做后处理、绘图、UI 更新与报警触发
        """
        frame_shape = frame_np.shape if hasattr(frame_np, 'shape') else 'Unknown'
        # self.log_message.emit(f"DEBUG: Received inference result for video {self.video_id}, frame shape={frame_shape}")
        h, w = frame_np.shape[:2]
        if (self.postproc.area_boxes == []) and (self.view_index is not None) and not self._area_loaded:
            boxes, log_msg = load_area_for_view(self.view_index, w, h)
            self.postproc.set_area_boxes(boxes)
            self.log_message.emit(log_msg)
            self._area_loaded = True
        elif (self.view_index is not None) and not self._area_loaded and not self._view_not_found_logged:
            # 如果区域框为空但view_index存在，说明找不到对应视角
            self.log_message.emit(f"警告：找不到视角 {self.view_index} 的检测区域配置")
            self._view_not_found_logged = True
            self._area_loaded = True  # 防止重复尝试加载
        structured = self.postproc.parse_raw_results(raw_results)
        # 更新状态并判断是否需要报警
        need_alert, info = self.postproc.update_state_and_check_alert(structured)

        for name, results in structured.items():
                self.detection_result.emit(name, results)

        annotated = self.drawer.draw(
            frame_np,
            info.get("all_results", {}),
            info.get("danger_detected", {}),
            info.get("danger_boxes", {}),
            area_boxes=self.postproc.area_boxes,
            alert_active=need_alert,
            alert_remaining=0.0,
        )

        # 发送 UI 信号
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        self.proc_frame_ready.emit(qimg.copy())

        # 如果需要报警并且是新告警，则发送异步邮件并触发 alert_message
        if need_alert:
            alert_parts = info.get("alert_parts", [])
            if alert_parts:
                specific_msg = "、".join(alert_parts)
                msg = f"检测到异常：{specific_msg}！"
            else:
                msg = "检测到异常！"
            self.alert_message.emit(msg)
            try:
                self.postproc.send_alert_async(self.video_name, msg, annotated, frame_np, email=self.alert_email)
            except Exception:
                pass


    @pyqtSlot(object, int)
    def attach_video_source(self, capture_manager, video_id):
        """
        附加视频源到 worker
        capture_manager: VideoCaptureManager 实例
        video_id: 视频流唯一标识
        """
        self.capture_manager = capture_manager
        self.video_id = video_id
    
        self.scheduler = Scheduler(target_fps=30.0, parent=self)
        self.scheduler.tick.connect(self.on_tick)
        self.scheduler.start()

        self.log_message.emit(f"视频源已附加: {video_id}")

    @pyqtSlot()
    def on_tick(self):
        # print("[TICK]", time.time(), "paused=", self._paused)
        if not self._running:
            return
        if self._stopped or self._paused:
            return
        # ⭐ 移除背压逻辑：即使推理线程忙碌，也将帧添加到队列等待处理
        # if self.inference_thread.is_busy():
        #     # self.log_message.emit(f"DEBUG: Inference thread is busy, skipping frame for video {self.video_id}")
        #     return
        frame = self.capture_manager.get_latest_frame(self.video_id)
        if frame is None:
            # self.log_message.emit(f"DEBUG: No frame obtained from buffer for video {self.video_id}")
            return
        frame_shape = frame.shape if hasattr(frame, 'shape') else 'Unknown'
        # 检查邮箱是否发生变化
        if self.alert_email != self._current_email:
            old_email = self._current_email if self._current_email else "未设置"
            new_email = self.alert_email if self.alert_email else "未设置"
            self.log_message.emit(f"报警邮箱已更新: {old_email} → {new_email}")
            self._current_email = self.alert_email
            # 更新postproc中的邮箱信息
            self.postproc.alert_email = self.alert_email
        # 只在第一次处理视频流时显示邮箱信息
        elif not self._email_info_logged:
            if self.alert_email:
                self.log_message.emit(f"当前视频流报警邮箱: {self.alert_email}")
            else:
                self.log_message.emit("当前视频流未设置报警邮箱，将使用默认管理员邮箱")
            self._email_info_logged = True
        # self.log_message.emit(f"DEBUG: Got frame from buffer for video {self.video_id}, shape={frame_shape}")
        self.inference_thread.add_task(frame)
        # self.log_message.emit(f"DEBUG: Sent frame to inference thread for video {self.video_id}")

    @pyqtSlot()
    def start(self):
        self._running = True
        self.scheduler.start()
        self.log_message.emit("MultiDetectorWorker 开始处理视频流")

    @pyqtSlot()
    def pause(self):
        self._paused = True
        self.inference_thread._enable = False
        # 移除日志输出，由MainController统一记录

    @pyqtSlot()
    def resume(self):
        self._paused = False
        self.inference_thread._enable = True
        # 移除日志输出，由MainController统一记录

    @pyqtSlot()
    def stop(self):
        """停止所有子模块"""
        self._running = False
        self.scheduler.stop()  # 不再周期性触发on_tick
        self.inference_thread.stop()
        self.log_message.emit("MultiDetectorWorker 已停止")
        
    @pyqtSlot(dict, dict, dict)
    def update_models(self, enabled, confidence, thresholds):
        """更新模型配置"""
        self._paused = True  # 先暂停当前推理
        self.inference_thread._enable = False
        for name, cfg in self.models.items():
            cfg.enabled = enabled.get(name, cfg.enabled)
            cfg.conf_threshold = confidence.get(name, cfg.conf_threshold)
            cfg.frame_threshold = thresholds.get(name, cfg.frame_threshold)
        self._paused = False  # 恢复推理
        self.inference_thread._enable = True



