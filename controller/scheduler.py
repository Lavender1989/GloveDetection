# Scheduler 负责周期性从 grabber 拉帧并放入 frame_queue（只保留最新），保证实时性

import time
import queue
from typing import Callable
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

# 改成QObject + QTimer
class Scheduler(QObject):
    tick = pyqtSignal()

    def __init__(self, target_fps=30.0, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / target_fps))
        self.timer.timeout.connect(self.tick)

    def start(self):
        self.timer.start()

    def stop(self):
        self.timer.stop()


# class Scheduler:
#     """
#     定时拉帧调度器
#     - grab_func: 无参函数，返回一帧 numpy 或 None
#     - out_queue: 用于放帧的 queue.Queue 实例（放入 (frame_id, frame_np)）
#     - target_fps: 期望的调度频率
#     """
#     def __init__(self, grab_func: Callable, out_queue: queue.Queue, target_fps: float = 10.0):
#         self.grab_func = grab_func
#         self.out_queue = out_queue
#         self.interval = 1.0 / max(0.1, target_fps)
#         self._running = False
#         self._thread = None
#         self._frame_id = 0

#     def start(self):
#         """启动独立线程执行定时调度"""
#         import threading
#         if self._running:
#             return
#         self._running = True
#         self._thread = threading.Thread(target=self._loop, daemon=True)
#         self._thread.start()

#     def stop(self):
#         self._running = False
#         if self._thread:
#             self._thread.join(timeout=1.0)

#     def _loop(self):
#         while self._running:
#             start = time.time()
#             frame = None
#             try:
#                 frame = self.grab_func()
#             except Exception:
#                 frame = None
#             if frame is not None:
#                 try:
#                     # 非阻塞放入，若满则丢旧（保持实时）
#                     try:
#                         self.out_queue.put_nowait((self._frame_id, frame))
#                         self._frame_id += 1
#                     except queue.Full:
#                         try:
#                             # 丢弃最旧的并放入最新
#                             self.out_queue.get_nowait()
#                             self.out_queue.put_nowait((self._frame_id, frame))
#                             self._frame_id += 1
#                         except Exception:
#                             pass
#                 except Exception:
#                     pass
#             elapsed = time.time() - start
#             to_sleep = self.interval - elapsed
#             if to_sleep > 0:
#                 time.sleep(to_sleep)
