# multi_detector_worker.py
import threading
import time
import queue
import os
import cv2
from typing import Dict, List, Optional
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker
from PyQt6.QtGui import QImage
from uuid import uuid4
from datetime import datetime

import torch

# 导入本地模块
from inference_thread import InferenceThread
from region_loader import load_area_for_view
from model.email_sender import EmailSender

# 你原先的类型/类 (DetectionResult, DetectionHistory, DetectionModel)
# 我们假定这些类保留原样并可从原文件导入；若不存在，把你原本定义放在同一文件或导入路径。
# from model.detection_types import DetectionResult, DetectionHistory, DetectionModel
# 为兼容，我将在使用位置假设这些都存在（你之前提供的代码片段包含这些定义）。


class MultiDetectorWorker(QObject):
    proc_frame_ready = pyqtSignal(QImage)
    log_message = pyqtSignal(str)
    alert_message = pyqtSignal(str)

    def __init__(self,
                 models_config: Dict[str, Dict],
                 video_name: str,
                 view_index: int,
                 alert_email: str,
                 capture_manager=None,
                 parent=None):
        """
        capture_manager: VideoCaptureManager 实例（必须实现 get_frame() 或 get_latest()）
        """
        super().__init__(parent)
        self._mutex = QMutex()

        # capture manager (生产帧)
        self.capture_manager = capture_manager

        # Models: name -> DetectionModel instance (assume user uses your DetectionModel)
        self.models: Dict[str, 'DetectionModel'] = {}
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

        self.enabled_models: Dict[str, bool] = {name: True for name in self.models.keys()}

        # status and alert variables (kept same)
        self.consecutive_danger_frames: Dict[str, int] = {name: 0 for name in self.models.keys()}
        self.alert_active = False
        self.alert_start_time = 0
        self.ALERT_DISPLAY_SECONDS = 5
        self.show_ui = True

        self.detection_histories: Dict[str, 'DetectionHistory'] = {}

        self.video_name = video_name
        self.view_index = view_index
        self.alert_email = alert_email
        self.email_sender = EmailSender()

        # Region data (loaded lazily when we have width/height)
        self.area_boxes: List[List[float]] = []
        self.area_loaded = False
        self.width = None
        self.height = None

        # Internal queues and threads
        self.frame_queue = queue.Queue(maxsize=6)   # buffer for scheduler
        self.scheduler_running = True

        # Inference thread (single worker)
        self.inference_thread = InferenceThread(self.models, self._mutex, queue_maxsize=4)
        self.inference_thread.inference_done.connect(self._handle_inference_result)
        self.inference_thread.inference_error.connect(self.log_message.emit)
        self.inference_thread.start()

        # Scheduler thread: pops from frame_queue and forwards to inference_thread
        self.scheduler_thread = threading.Thread(target=self._inference_scheduler, daemon=True)
        self.scheduler_thread.start()

        # Grab thread: 从 capture_manager 拉帧放入 frame_queue（若未提供 capture_manager，则外部仍可调用 process_frame）
        self.grab_running = False
        if self.capture_manager is not None:
            self.grab_running = True
            self.grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
            self.grab_thread.start()

        # bookkeeping
        self.current_frame_id = 0

        # log GPU info
        try:
            cuda_available = torch.cuda.is_available()
            self.log_message.emit(f"CUDA可用: {cuda_available}")
            if cuda_available:
                self.log_message.emit(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

        self.log_message.emit(f"MultiDetectorWorker 初始化完成，模型: {list(self.models.keys())}")

    # ------------- input side -------------
    def process_frame(self, frame: 'np.ndarray'):
        """
        旧接口：外部主动把帧塞进来（向后兼容）
        我们会把它放入frame_queue
        """
        try:
            self.frame_queue.put_nowait((self.current_frame_id, frame))
            self.current_frame_id += 1
        except queue.Full:
            try:
                # 丢旧帧保持实时性
                _ = self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((self.current_frame_id, frame))
                self.current_frame_id += 1
            except Exception:
                pass

    def _grab_loop(self):
        """
        从 capture_manager 有节奏地拉最新帧并放入 frame_queue。
        若 frame_queue 满则丢旧帧，保证实时性。
        """
        # pull interval: try to respect capture_manager's source fps if available
        pull_dt = 1.0 / 15.0  # default 15hz
        # try discover fps from capture_manager.reader if exists
        try:
            reader = getattr(self.capture_manager, 'reader', None)
            if reader is not None:
                fps = getattr(reader, 'fps', None)
                if fps and fps > 1 and fps < 120:
                    pull_dt = 1.0 / fps
        except Exception:
            pass

        while self.grab_running:
            try:
                # Prefer get_latest if capture_manager supports it
                frame = None
                if hasattr(self.capture_manager, 'get_latest'):
                    frame = self.capture_manager.get_latest()
                elif hasattr(self.capture_manager, 'get_frame'):
                    frame = self.capture_manager.get_frame(timeout=0.05)
                else:
                    # fallback: no capture manager API
                    frame = None

                if frame is None:
                    time.sleep(0.005)
                    continue

                # put into frame_queue (non-blocking)
                try:
                    self.frame_queue.put_nowait((self.current_frame_id, frame))
                    self.current_frame_id += 1
                except queue.Full:
                    # drop oldest and push new
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((self.current_frame_id, frame))
                        self.current_frame_id += 1
                    except Exception:
                        pass

                # set width/height on first valid frame if unknown
                if not self.area_loaded and frame is not None:
                    try:
                        h, w = frame.shape[:2]
                        if (self.width != w) or (self.height != h):
                            self.width, self.height = w, h
                            self._reload_area()
                    except Exception:
                        pass

            except Exception as e:
                # log and continue
                try:
                    self.log_message.emit(f"grab_loop error: {e}")
                except Exception:
                    pass
                time.sleep(0.01)
            finally:
                time.sleep(pull_dt)

    # ------------- scheduler side -------------
    def _get_latest_from_frame_queue(self):
        """
        从 frame_queue 中取最新帧（如果队列有多个元素只取最后一个）。
        返回 (frame_id, frame_np) 或 (None, None)。
        """
        item = None
        try:
            item = self.frame_queue.get(block=True, timeout=0.1)
        except queue.Empty:
            return None, None

        # drain to latest
        while True:
            try:
                nxt = self.frame_queue.get_nowait()
                item = nxt
            except queue.Empty:
                break
        return item

    def _inference_scheduler(self):
        """
        从frame_queue 中取最新帧，检查是否有启用模型并提交给 inference_thread.
        """
        while self.scheduler_running:
            try:
                item = self._get_latest_from_frame_queue()
                if item is None or item[0] is None:
                    continue
                frame_id, frame_np = item

                # if no models enabled skip
                with QMutexLocker(self._mutex):
                    any_enabled = any(self.enabled_models.values())

                if not any_enabled:
                    continue

                # Submit to inference thread (non-blocking strategy inside inference_thread handles overflow)
                ok = self.inference_thread.add_task(frame_np, {'frame_id': frame_id})
                if not ok:
                    try:
                        self.log_message.emit("推理队列已满，丢弃帧")
                    except Exception:
                        pass

            except Exception as e:
                try:
                    self.log_message.emit(f"调度线程异常: {e}")
                except Exception:
                    pass
                time.sleep(0.01)

    # ------------- result handling (分离函数) -------------
    def _handle_inference_result(self, frame_np, raw_results: Dict[str, object]):
        """
        作为 InferenceThread.inference_done 的 slot 被调用（主线程上下文）。
        这里把 raw_results 转成 DetectionResult 列表，应用触发/历史/报警逻辑，并发 UI 信号。
        """
        try:
            # Convert raw results -> structured detections (model_name -> List[DetectionResult])
            structured = self._process_det_results(raw_results)

            # Update state, counters, history, and maybe trigger alert
            should_alert, alert_info = self._update_state_and_alert(structured)

            # Draw on frame (returns annotated frame)
            annotated = self._draw_ui(frame_np.copy(), structured, alert_info['danger_detected'], alert_info['danger_boxes'])

            # emit UI update
            if self.show_ui:
                try:
                    h, w = annotated.shape[:2]
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
                    self.proc_frame_ready.emit(qimg.copy())
                except Exception as e:
                    self.log_message.emit(f"UI update error: {e}")

            # If alert needs asynchronous email send, handle it
            if should_alert and not self.alert_active:
                # use same logic as original: save images and spawn email thread
                self._trigger_alert(annotated, frame_np, alert_info)
        except Exception as e:
            try:
                self.log_message.emit(f"_handle_inference_result error: {e}")
            except Exception:
                pass

    def _process_det_results(self, raw_results: Dict[str, object]) -> Dict[str, List['DetectionResult']]:
        """
        将 raw_results（YOLO 返回的 objects）转为 DetectionResult 列表并筛选 target_classes & conf
        """
        structured = {}
        for name, model_cfg in self.models.items():
            structured[name] = []
            y = raw_results.get(name, None)
            if y is None:
                continue
            try:
                boxes = y.boxes.xyxy.cpu().numpy() if hasattr(y.boxes, "xyxy") else []
                confs = y.boxes.conf.cpu().numpy() if hasattr(y.boxes, "conf") else []
                cls_ids = y.boxes.cls.cpu().numpy() if hasattr(y.boxes, "cls") else []
                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    cls_name = y.names[int(cls_id)]
                    if cls_name in model_cfg.target_classes and conf >= model_cfg.conf_threshold:
                        structured[name].append(DetectionResult(model_cfg.detection_type, cls_name, float(conf), box.tolist()))
            except Exception as e:
                try:
                    self.log_message.emit(f"process_det_results error for {name}: {e}")
                except Exception:
                    pass
        return structured

    def _update_state_and_alert(self, structured_results: Dict[str, List['DetectionResult']]):
        """
        应用触发模式(area/any)、连续帧计数、历史过滤。
        返回 (should_alert, alert_info_dict)
        alert_info_dict 包含 danger_detected, danger_boxes, all_results
        """
        now = time.time()
        danger_detected = {k: False for k in self.models.keys()}
        danger_boxes = {k: [] for k in self.models.keys()}

        # Apply trigger mode
        for name, model_cfg in self.models.items():
            if not self.enabled_models.get(name, False):
                continue
            detections = structured_results.get(name, []) or []
            if model_cfg.trigger_mode == 'area':
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

        # update consecutive counts
        for name in self.models.keys():
            if not self.enabled_models.get(name, False):
                self.consecutive_danger_frames[name] = 0
                continue
            if danger_detected.get(name, False):
                self.consecutive_danger_frames[name] += 1
            else:
                self.consecutive_danger_frames[name] = 0

        # determine if should alert
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

        # historical new detection check
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

        return need_new_alert, {
            'danger_detected': danger_detected,
            'danger_boxes': danger_boxes,
            'new_detections': new_detections,
            'all_results': structured_results,
            'alert_parts': alert_parts
        }

    def _draw_ui(self, frame: 'np.ndarray', all_results: Dict[str, List['DetectionResult']], danger_detected: Dict[str, bool], danger_boxes: Dict[str, List['DetectionResult']]) -> 'np.ndarray':
        """
        把检测结果绘制到帧上。保留你原来的样式，但简化代码组织。
        """
        annotated = frame
        h, w = annotated.shape[:2]
        font_scale = 1.0
        line_thickness = max(1, int(font_scale * 2))

        # draw areas
        for box in self.area_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, "area", (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # draw detections
        for name, dets in all_results.items():
            if not self.enabled_models.get(name, False):
                continue
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
                    cv2.putText(annotated, f"{label_prefix} {det.confidence:.2f}", (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # alert banner
        if self.alert_active:
            remaining = max(0, self.ALERT_DISPLAY_SECONDS - (time.time() - self.alert_start_time))
            cv2.putText(annotated, f"ALERT ({remaining:.1f}s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        return annotated

    def _trigger_alert(self, annotated_frame, original_frame, alert_info):
        """
        创建报警图片/文件并异步发送邮件（与原逻辑保持一致）
        """
        try:
            # mark alert active
            self.alert_active = True
            self.alert_start_time = time.time()
            parts = alert_info.get('alert_parts', [])
            msg = "检测到" + "和".join(parts) + "！"
            self.alert_message.emit(msg)
            self.log_message.emit(f"[报警] {msg}")

            # save frames
            try:
                alert_images_dir = os.path.join(os.path.dirname(__file__), '..', 'alert_images')
                os.makedirs(alert_images_dir, exist_ok=True)
                unique_id = uuid4().hex
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                original_path = os.path.join(alert_images_dir, f"alert_{ts}_{unique_id}_original.jpg")
                annotated_path = os.path.join(alert_images_dir, f"alert_{ts}_{unique_id}_annotated.jpg")
                cv2.imwrite(original_path, original_frame)
                cv2.imwrite(annotated_path, annotated_frame)
            except Exception as e:
                self.log_message.emit(f"保存报警图片失败: {e}")
                original_path = annotated_path = None

            # Asynchronous email send
            def _send():
                try:
                    self.email_sender.send_alert_email(self.video_name, msg, annotated_frame, self.alert_email, original_frame, None)
                except Exception as e:
                    self.log_message.emit(f"发送报警邮件失败: {e}")
                finally:
                    # cleanup temp images
                    for p in [original_path, annotated_path]:
                        if p and os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass

            th = threading.Thread(target=_send, daemon=True)
            th.start()

            # reset consecutive counts for all models to avoid repeated immediate alerts
            for k in self.consecutive_danger_frames.keys():
                self.consecutive_danger_frames[k] = 0

        except Exception as e:
            try:
                self.log_message.emit(f"_trigger_alert error: {e}")
            except Exception:
                pass

    # ------------- utils -------------
    def _reload_area(self):
        try:
            boxes, log_msg = load_area_for_view(self.view_index, self.width, self.height)
            self.area_boxes = boxes
            self.area_loaded = True
            self.log_message.emit(log_msg)
            self.log_message.emit(f"区域加载: {len(boxes)}")
        except Exception as e:
            self.log_message.emit(f"区域加载失败: {e}")

    def _box_fully_contains(self, container_box: List[float], inner_box: List[float]) -> bool:
        x1, y1, x2, y2 = [float(x) for x in container_box]
        x3, y3, x4, y4 = [float(x) for x in inner_box]
        return (x3 >= x1) and (y3 >= y1) and (x4 <= x2) and (y4 <= y2)

    # ------------- lifecycle -------------
    def stop(self):
        # stop grab loop
        self.grab_running = False
        try:
            if hasattr(self, 'grab_thread'):
                self.grab_thread.join(timeout=1.0)
        except Exception:
            pass

        # stop scheduler
        self.scheduler_running = False
        try:
            if hasattr(self, 'scheduler_thread'):
                self.scheduler_thread.join(timeout=1.0)
        except Exception:
            pass

        # stop inference thread
        try:
            if hasattr(self, 'inference_thread'):
                self.inference_thread.stop()
        except Exception:
            pass

        self.log_message.emit("MultiDetectorWorker stopped")
