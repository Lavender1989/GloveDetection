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
from .types import DetectionModel, ModelManager
import threading


class InferenceThread(QThread):
    inference_done = pyqtSignal(object, object)  # frame_np, raw_results
    inference_error = pyqtSignal(str)

    def __init__(self, models: Dict[str, 'DetectionModel'], queue_maxsize: int = 10, parent=None):
        super().__init__(parent)
        self.models = models
        self._running = True
        self._q = queue.Queue(maxsize=queue_maxsize)  # 增加队列大小，提高缓存能力
        self._busy = False   # 忙碌态
        self._enable = True  # 推理线程是否启用
        
        # 获取全局模型管理器实例
        self.model_manager = ModelManager()

        try:
            if torch.cuda.is_available():
                torch.cuda.init()
        except Exception:
            pass

    @pyqtSlot(object)
    def add_task(self, frame_np: np.ndarray) -> bool:
        """非阻塞地添加任务；若队列满则尝试等待一小段时间再入队（减少丢帧）"""
        
        if not frame_np.any():
            return False
        
        try:
            self._q.put_nowait(frame_np)
            return True
        except queue.Full:
            # 丢弃最旧，保留新帧
            try:
                old_frame = self._q.get_nowait()
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
        discard_count = 0
        while True:
            try:
                nxt = self._q.get_nowait()
                item = nxt
                discard_count += 1
            except queue.Empty:
                break
        
        return item

    def stop(self):
        """停止线程（外部调用）"""
        self._running = False  # 不再消费帧
        
        # 等待线程结束
        self.quit()
        self.wait()
        
        # 清除GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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
            
            self._busy = True   # ⭐ 推理开始
            raw_results = {}
            
            # 使用全局推理锁确保模型推理的线程安全
            with self.model_manager.inference_global_lock:
                try:
                    for name, model_cfg in self.models.items():
                        if not model_cfg.enabled:
                            raw_results[name] = None
                            continue
                        try:
                            with torch.no_grad():
                                # 优化推理参数，减少GPU内存使用
                                y = model_cfg.model(
                                    frame,
                                    conf=model_cfg.conf_threshold,
                                    device=model_cfg.device,  # 使用模型实际加载的设备
                                    imgsz=320,  # 降低输入分辨率以减少内存使用
                                    half=model_cfg.device == 'cuda',  # 只有在GPU上才使用半精度
                                    batch=1,    # 批量大小为1
                                    verbose=False,
                                    # 减少数据增强，降低内存和计算需求
                                    augment=False,
                                    # 不保存轨迹，减少内存使用
                                    save=False,
                                    # 使用更小的iou阈值
                                    iou=0.45
                                )[0]
                            raw_results[name] = y
                        except Exception as e:
                            raw_results[name] = None
                            self.inference_error.emit(f"推理错误 ({name}): {str(e)}")
                finally:
                    self._busy = False   # ⭐ 推理结束
                    
                    # 推理完成后清理临时GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            
            self.inference_done.emit(frame, raw_results)
            
            # 清除GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def is_busy(self):
        return self._busy