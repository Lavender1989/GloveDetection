""" 
# GPU 推理线程：只做模型前向推理，不做后处理
# 将 raw model 返回值（ultralytics 的返回对象）发回主线程供 CPU 端处理
# 注：使用 PyQt 的 QThread + signals，将结果发回主线程处理 UI/报警等逻辑
"""

from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
import queue
import time
import numpy as np
from typing import Dict, Tuple, Any
import torch
from .types import DetectionModel


class InferenceThread(QThread):
    inference_done = pyqtSignal(object, object)  # frame_np, raw_results
    inference_error = pyqtSignal(str)

    def __init__(self, models: Dict[str, 'DetectionModel'], queue_maxsize: int = 4, parent=None):
        super().__init__(parent)
        self.models = models
        self._running = True
        self._q = queue.Queue(maxsize=queue_maxsize)
        self._busy = False   # 忙碌态
        self._enable = True  # 推理线程是否启用

        try:
            if torch.cuda.is_available():
                torch.cuda.init()
        except Exception:
            pass

    @pyqtSlot(object)
    def add_task(self, frame_np: np.ndarray) -> bool:
        """非阻塞地添加任务；若队列满则尝试丢旧帧并入队（保持实时性）"""
        if frame_np is None:
            return False
        
        if not frame_np.any():
            return False
        try:
            self._q.put_nowait(frame_np)
            return True
        except queue.Full:
            # 丢弃最旧，保留新帧
            try:
                self._q.get_nowait()
                self._q.put_nowait(frame_np)
                return True
            except Exception:
                return False

    def _pop_latest(self) -> Tuple[np.ndarray, dict]:
        """
        从队列中取最新任务，丢弃队列中旧任务以减少延迟
        """
        item = None
        try:
            # block for a short time to wait for at least one item
            item = self._q.get(block=True, timeout=0.1)
        except queue.Empty:
            return None, None

        # 把队列中剩余的都取出，只保留最后一个（最新）
        while True:
            try:
                nxt = self._q.get_nowait()
                item = nxt
            except queue.Empty:
                break
        return item

    def stop(self):
        """停止线程（外部调用）"""
        self._running = False  # 不再消费帧
        self.quit()
        self.wait()

    def run(self):
        while self._running:
            if not self._enable:
                time.sleep(0.1)
                continue
            try:
                frame = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame is None or not frame.any():
                self._busy = False
                continue

            raw_results = {}
            self._busy = True   # ⭐ 推理开始
            for name, model_cfg in self.models.items():
                if not model_cfg.enabled:
                    raw_results[name] = None
                    continue
                try:
                    y = model_cfg.model(
                        frame,
                        conf=model_cfg.conf_threshold,
                        device='cuda',
                        imgsz=320,
                        half=True,
                        verbose=False
                    )[0]
                    raw_results[name] = y
                except Exception as e:
                    raw_results[name] = None
                    self.inference_error.emit(str(e))
                    
            self._busy = False   # ⭐ 推理结束
            self.inference_done.emit(frame, raw_results)

    def is_busy(self):
        return self._busy
