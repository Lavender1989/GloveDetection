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
            print(f"[DEBUG] InferenceThread.add_task: 收到空帧")
            return False
        
        if not frame_np.any():
            print(f"[DEBUG] InferenceThread.add_task: 收到无效帧（全黑或全白）")
            return False
        
        print(f"[DEBUG] InferenceThread.add_task: 收到有效帧，形状: {frame_np.shape}")
        try:
            self._q.put_nowait(frame_np)
            print(f"[DEBUG] InferenceThread.add_task: 帧已入队，当前队列大小: {self._q.qsize()}")
            return True
        except queue.Full:
            # 丢弃最旧，保留新帧
            try:
                old_frame = self._q.get_nowait()
                print(f"[DEBUG] InferenceThread.add_task: 队列已满，丢掉旧帧")
                self._q.put_nowait(frame_np)
                print(f"[DEBUG] InferenceThread.add_task: 新帧已入队，当前队列大小: {self._q.qsize()}")
                return True
            except Exception as e:
                print(f"[DEBUG] InferenceThread.add_task: 入队失败: {e}")
                return False

    def _pop_latest(self) -> Tuple[np.ndarray, dict]:
        """
        从队列中取最新任务，丢弃队列中旧任务以减少延迟
        """
        item = None
        try:
            # block for a short time to wait for at least one item
            item = self._q.get(block=True, timeout=0.1)
            print(f"[DEBUG] InferenceThread._pop_latest: 从队列获取第一个帧")
        except queue.Empty:
            print(f"[DEBUG] InferenceThread._pop_latest: 队列空，超时")
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
        
        if item is not None:
            print(f"[DEBUG] InferenceThread._pop_latest: 获取到最新帧，形状: {item.shape}，丢弃了 {discard_count} 个旧帧")
        else:
            print(f"[DEBUG] InferenceThread._pop_latest: 未获取到有效帧")
        return item

    def stop(self):
        """停止线程（外部调用）"""
        self._running = False  # 不再消费帧
        self.quit()
        self.wait()

    def run(self):
        print(f"[DEBUG] InferenceThread.run: 推理线程开始运行")
        inference_count = 0
        while self._running:
            if not self._enable:
                print(f"[DEBUG] InferenceThread.run: 推理线程已禁用，等待...")
                time.sleep(0.1)
                continue
            try:
                frame = self._q.get(timeout=0.1)
                print(f"[DEBUG] InferenceThread.run: 从队列获取帧")
            except queue.Empty:
                # print(f"[DEBUG] InferenceThread.run: 队列为空，继续等待...")
                continue
                
            if frame is None or not frame.any():
                print(f"[DEBUG] InferenceThread.run: 收到空帧或无效帧")
                self._busy = False
                continue
            
            inference_count += 1
            print(f"[DEBUG] InferenceThread.run: 第 {inference_count} 次推理开始，帧形状: {frame.shape}")

            raw_results = {}
            self._busy = True   # ⭐ 推理开始
            for name, model_cfg in self.models.items():
                if not model_cfg.enabled:
                    print(f"[DEBUG] InferenceThread.run: 模型 {name} 已禁用，跳过")
                    raw_results[name] = None
                    continue
                try:
                    print(f"[DEBUG] InferenceThread.run: 开始模型 {name} 推理")
                    y = model_cfg.model(
                        frame,
                        conf=model_cfg.conf_threshold,
                        device='cuda',
                        imgsz=320,
                        half=True,
                        verbose=False
                    )[0]
                    raw_results[name] = y
                    print(f"[DEBUG] InferenceThread.run: 模型 {name} 推理完成")
                except Exception as e:
                    raw_results[name] = None
                    print(f"[DEBUG] InferenceThread.run: 模型 {name} 推理失败: {e}")
                    self.inference_error.emit(str(e))
                    
            self._busy = False   # ⭐ 推理结束
            print(f"[DEBUG] InferenceThread.run: 第 {inference_count} 次推理完成，发送结果信号")
            self.inference_done.emit(frame, raw_results)

    def is_busy(self):
        return self._busy
