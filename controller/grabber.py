# Grabber 负责从外部 VideoCaptureManager 读取最新帧并交付给上层
# 这里不直接创建 VideoCaptureManager，而是接收其实例（保持解耦）
# 已弃用


import time
from typing import Optional

class FrameGrabber:
    """从 VideoCaptureManager 拉取帧的适配器"""
    def __init__(self, capture_manager):
        """
        capture_manager: VideoCaptureManager 实例 get_latest: 获取后进队列 get: 获取先进队列的
        """
        self.capture_manager = capture_manager
        # 检查是否为 MultiDetectorWorker 实例
        if hasattr(capture_manager, 'capture_manager') and hasattr(capture_manager, 'video_id'):
            self.worker = capture_manager
            self.capture_manager = capture_manager.capture_manager
            self.video_id = capture_manager.video_id
        else:
            self.worker = None
            self.video_id = getattr(capture_manager, 'video_id', None)  # 尝试获取视频ID

    def grab(self, timeout: float = 0.05) -> Optional['np.ndarray']:
        """
        获取一帧：直接从capture_manager获取帧
        """
        try:
            if self.capture_manager is not None and self.video_id is not None:
                # 优先获取最新帧
                if hasattr(self.capture_manager, 'get_latest_frame'):
                    # print(f"[DEBUG] FrameGrabber: 获取到一帧({self.video_id})")
                    return self.capture_manager.get_latest_frame(self.video_id)
                elif hasattr(self.capture_manager, 'get_frame'):
                    return self.capture_manager.get_frame(self.video_id, timeout=timeout)
            return None
        except Exception as e:
            print(f"FrameGrabber.grab 错误: {e}")
            return None