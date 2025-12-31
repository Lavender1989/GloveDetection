# -*- coding: utf-8 -*-
"""
跨平台视频捕获管理器
- 统一使用OpenCV CPU解码
- 提供统一的torch tensor接口
- 支持多路RTSP流管理
- 可配置缓冲策略
- 保留GStreamer硬件解码支持（作为可选功能）
"""

import platform
import time
import threading
import queue
import cv2
import os
from time import sleep
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker

# =============== 1. 后端选择器（保留GStreamer支持） ===============
class VideoBackendSelector:
    @staticmethod
    def is_jetson():
        # Jetson 会显示为 aarch64
        return platform.machine() == "aarch64"

# =============== 2. 线程安全 Buffer ===============
class FrameBuffer:
    # 单个视频流的帧缓冲
    def __init__(self, maxsize=5):
        self.q = queue.Queue(maxsize=maxsize)

    def push(self, frame):
        # print(f"[DEBUG] FrameBuffer.push: 收到帧，形状: {frame.shape}")
        if self.q.full():
            try:
                self.q.get_nowait()  # 丢掉最旧帧
                # print(f"[DEBUG] FrameBuffer.push: 队列已满，丢掉旧帧")
            except queue.Empty:
                pass
        self.q.put_nowait(frame)

    def get_latest(self):
        # LIFO 策略：获取最新帧(不要清空所有帧)
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def get(self):
        # 向后兼容旧代码，提供get方法
        return self.get_latest()

# =============== 3. 统一 VideoReader ===============
class VideoReader:
    def __init__(self, url, fps_limit=None, use_opencv_cuda=False, use_gstreamer=False):
        """
        统一使用CPU的OpenCV VideoReader
        
        参数：
        - url: 视频路径或RTSP地址
        - fps_limit: FPS限制
        - use_opencv_cuda: 兼容参数，不再使用
        - use_gstreamer: 是否尝试使用GStreamer硬件解码（仅在Jetson平台有效）
        """
        self.url = url
        self.last_frame_ts = 0.0  # 增加一个心跳判断是否有新帧到达
        self.is_file = not url.lower().startswith("rtsp://")  # 判断是不是rtsp视频
        self.fps_limit = fps_limit
        self._running = True
        
        # 尝试使用GStreamer硬件解码（仅在Jetson平台和use_gstreamer=True时）
        if use_gstreamer and VideoBackendSelector.is_jetson():
            try:
                # 检测视频编码格式
                is_h265 = False
                try:
                    # 尝试使用ffprobe检测视频编码格式
                    import subprocess
                    result = subprocess.run(
                        ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                         "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", 
                         self.url],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        codec_name = result.stdout.strip().lower()
                        is_h265 = codec_name == "hevc"
                except Exception:
                    # 如果ffprobe不可用或检测失败，基于文件扩展名猜测
                    if self.url.lower().endswith(".mp4"):
                        # 默认MP4可能是H.264或H.265，尝试H.265
                        is_h265 = True
                
                if is_h265:
                    # Jetson平台上的H.265视频使用GStreamer硬件解码管道
                    print(f"[INFO] 使用GStreamer硬件解码H.265视频: {self.url}")
                    # 构建GStreamer管道
                    if self.is_file:
                        # 本地文件
                        gst_pipeline = (
                            f"filesrc location={self.url} ! "
                            "qtdemux ! h265parse ! "
                            "nvv4l2decoder codec=h265 ! nvvidconv ! "
                            "video/x-raw, format=(string)BGRx ! "
                            "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
                        )
                    else:
                        # RTSP流
                        gst_pipeline = (
                            f"rtspsrc location={self.url} latency=0 ! "
                            "rtph265depay ! h265parse ! "
                            "nvv4l2decoder codec=h265 ! nvvidconv ! "
                            "video/x-raw, format=(string)BGRx ! "
                            "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
                        )
                    
                    # 使用GStreamer管道创建VideoCapture
                    self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                    if self.cap.isOpened():
                        print("[INFO] GStreamer硬件解码初始化成功")
                        # 获取视频 FPS
                        fps = self.cap.get(cv2.CAP_PROP_FPS)
                        self.fps_dt = 1.0 / (fps if 1 < fps <= 120 else 30)
                        return
                    else:
                        print("[WARNING] GStreamer硬件解码管道打开失败，回退到OpenCV")
            except Exception as e:
                print(f"[WARNING] GStreamer硬件解码初始化失败: {e}")
        
        # 统一使用基本的OpenCV VideoCapture（默认方式）
        try:
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.url}")
        except Exception as e:
            print(f"[WARNING] OpenCV初始化失败: {e}")
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.url}")
        
        # 获取视频 FPS
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps_dt = 1.0 / (fps if 1 < fps <= 120 else 30)

    def read(self):
        """
        读取一帧视频
        返回: (ret, frame)
        ret: 是否成功
        frame: 视频帧（numpy数组）
        """
        ret, frame = self.cap.read()
        if self.fps_limit:
            time.sleep(1.0 / self.fps_limit)
        return ret, frame

    def release(self):
        """
        释放资源
        """
        self._running = False
        if hasattr(self, 'cap'):
            self.cap.release()

# =============== 4. VideoCaptureManager（多视频流管理器） ===============
class VideoCaptureManager(QObject):
    """
    多视频流管理器：管理多个视频源的捕获线程和缓冲
    """
    log_message = pyqtSignal(str)
    rtsp_disconnected = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._streams = {}  # {video_id: (capture_thread, buffer)}
        self._mutex = QMutex()
        self._running = True
        self._paused_streams = set()  # 暂停中的视频流ID
        
    def add_video_stream(self, video_id, video_url, fps_limit=None, use_opencv_cuda=False, use_gstreamer=False):
        """
        添加视频流
        video_id: 视频唯一标识
        video_url: RTSP地址或文件路径
        fps_limit: FPS限制（可选）
        use_opencv_cuda: 兼容参数，不再使用
        use_gstreamer: 是否尝试使用GStreamer硬件解码（仅在Jetson平台有效）
        创建buffer并传给reader
        """
        with QMutexLocker(self._mutex):
            if video_id in self._streams:
                self.log_message.emit(f"视频流 {video_id} 已存在")
                return False

            try:
                # 创建帧缓冲
                buffer = FrameBuffer(maxsize=5)
                
                # 创建并启动捕获线程
                capture_thread = threading.Thread(
                    target=self._capture_loop,
                    args=(video_id, video_url, buffer, fps_limit, use_opencv_cuda, use_gstreamer),
                    daemon=True
                )
                
                self._streams[video_id] = (capture_thread, buffer)
                capture_thread.start()
                self.log_message.emit(f"添加视频流成功: {video_id} -> {video_url}")
                return True
            except Exception as e:
                self.log_message.emit(f"添加视频流失败 {video_id}: {e}")
                return False

    def remove_video_stream(self, video_id):
        """
        移除视频流
        """
        with QMutexLocker(self._mutex):
            if video_id not in self._streams:
                return False
                
            try:
                self._streams.pop(video_id)
                # 线程会自动退出，因为是daemon线程
                self.log_message.emit(f"移除视频流: {video_id}")
                return True
            except Exception as e:
                self.log_message.emit(f"移除视频流失败 {video_id}: {e}")
                return False

    def get_video_buffer(self, video_id):
        """
        获取指定视频流的帧缓冲
        """
        with QMutexLocker(self._mutex):
            if video_id not in self._streams:
                return None
            return self._streams[video_id][1]

    def pause_video(self, video_id):
        """
        暂停指定视频流的捕获
        """
        with QMutexLocker(self._mutex):
            if video_id in self._streams:
                self._paused_streams.add(video_id)
                self.log_message.emit(f"视频流 {video_id} 已暂停")
    
    def resume_video(self, video_id):
        """
        恢复指定视频流的捕获
        """
        with QMutexLocker(self._mutex):
            if video_id in self._paused_streams:
                self._paused_streams.remove(video_id)
                self.log_message.emit(f"视频流 {video_id} 已恢复")

    def get_latest_frame(self, video_id):
        """
        获取指定视频流的最新帧
        如果buffer还没来得及push就会得到None
        """
        buffer = self.get_video_buffer(video_id)
        if buffer:
            return buffer.get_latest()
        return None

    def _capture_loop(self, video_id, video_url, buffer, fps_limit=None, use_opencv_cuda=False, use_gstreamer=False):
        """
        单个视频流的捕获循环
        统一使用OpenCV模型：
            capture_loop (线程)
            ↓
            reader.read()
            ↓
            buffer.push(frame)
        """
        try:
            # 创建视频读取器
            reader = VideoReader(video_url, fps_limit=fps_limit, use_opencv_cuda=use_opencv_cuda, use_gstreamer=use_gstreamer)
            fps_dt = 1.0 / fps_limit if fps_limit else getattr(reader, 'fps_dt', 1/30)
            
            last_t = 0
            reconnect_count = 0
            was_reconnecting = False # 是否需要重连
            frame_count = 0
            success_count = 0

            while self._running:
                # ⭐ 暂停检查
                if video_id in self._paused_streams:
                    time.sleep(0.02)
                    continue
                
                # 按FPS限速
                if fps_dt:
                    now = time.time()
                    if now - last_t < fps_dt:
                        time.sleep(min(fps_dt * 0.5, 0.01))
                        continue
                    last_t = now
                
                frame_count += 1
                
                # 使用OpenCV读取帧
                ret, frame = reader.read()
                if not ret or frame is None:
                    if reader.is_file:
                        # ⭐ 本地视频：播放完毕，需要循环播放
                        try:
                            reader.release()
                            reader = VideoReader(video_url, fps_limit=fps_limit, use_gstreamer=use_gstreamer)
                            last_t = 0
                            continue
                        except Exception as e:
                            time.sleep(1.0)
                            continue
                    else:
                        # RTSP重连
                        reconnect_count += 1
                        was_reconnecting = True  # 正在进行重连
                        self.log_message.emit(f"[视频 {video_id}] 连接丢失，正在重连... ({reconnect_count})")
                        time.sleep(1.0)
                        
                        # 重连机制
                        if reconnect_count > 5:
                            self.log_message.emit(f"[视频 {video_id}] 重连失败，放弃")
                            self.rtsp_disconnected.emit(video_id)
                            break
                        try:
                            reader.release()
                            reader = VideoReader(video_url, fps_limit=fps_limit, use_gstreamer=use_gstreamer)
                            self.log_message.emit(f"[视频 {video_id}] 重连成功")
                        except Exception as e:
                            self.log_message.emit(f"[视频 {video_id}] 重连失败: {e}")
                            time.sleep(1.0)
                    continue
                
                success_count += 1
                
                # 过滤全黑帧
                if not frame.any():
                    continue

                if was_reconnecting:
                    self.log_message.emit(f"[视频 {video_id}] 重连成功")
                    was_reconnecting = False  # 重连成功，重置标志
                    reconnect_count = 0
                
                buffer.push(frame)
                continue
                
        except Exception as e:
            self.log_message.emit(f"[视频 {video_id}] 捕获错误: {e}")
        finally:
            try:
                reader.release()
            except Exception:
                pass

    def stop(self):
        """
        停止所有视频流
        """
        self._running = False
        with QMutexLocker(self._mutex):
            self._streams.clear()

if __name__ == "__main__":
    """
    测试代码：
    VideoReader能否正常打开视频文件
    FrameBuffer + VideoCaptureManager能否持续读入numpy帧
    """

    buffer = FrameBuffer()
    cap_manager = VideoCaptureManager()
    video_id = "test_video"
    # video_url = r"D:\detect_video\original_Video\no_wearing_gloves\0911_1.mp4"
    video_url = input("输入测试视频路径")
    cap_manager.add_video_stream(video_id, video_url)

    print("Capture manager started. Press Ctrl+C to stop.")

    while True:
        frame = cap_manager.get_latest_frame(video_id)
        if frame is not None:
            cv2.imshow("Test Frame", frame)

        # 必须加，否则窗口不刷新
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    sleep(0.01)
    cap_manager.stop()
    cv2.destroyAllWindows()