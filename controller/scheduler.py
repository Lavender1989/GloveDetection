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

