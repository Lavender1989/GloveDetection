# -*- coding: utf-8 -*-
"""
跨平台视频捕获管理器
- Windows平台：使用OpenCV CPU解码
- Jetson平台：直接使用OpenCV-CUDA解码
- 提供统一的torch tensor接口
- 支持多路RTSP流管理
- 可配置缓冲策略
"""

import platform
import time
import threading
import queue
import cv2
import os
from time import sleep
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker

# =============== 1. 自动选择解码后端 ===============
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

# =============== 3. 跨平台 VideoReader ===============
class VideoReader:
    def __init__(self, url, fps_limit=None, use_opencv_cuda=False):
        """
        跨平台 VideoReader
        Windows: OpenCV
        Jetson: 
            - 默认: OpenCV-CUDA
        """
        self.url = url
        self.last_frame_ts = 0.0  # 增加一个心跳判断是否有新帧到达
        self.is_file = not url.lower().startswith("rtsp://")  # 判断是不是rtsp视频
        self.fps_limit = fps_limit
        self._running = True
        
        # 检查是否在Jetson平台上
        if VideoBackendSelector.is_jetson():
            # Jetson平台默认使用OpenCV-CUDA
            # 设置环境变量禁用GStreamer并启用CUDA加速
            os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'  # 禁用GStreamer优先级
            os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '100'  # 提高FFMPEG优先级
            
            # 根据视频URL判断可能的编码格式
            # 这里可以根据需要扩展，目前支持H.264和H.265
            if self.url.lower().endswith('.mp4') or 'rtsp://' in self.url.lower():
                # 尝试自动检测或使用通用配置支持多种编码
                # 对于H.265(HEVC)使用hevc_cuvid解码器
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'video_codec;hevc_cuvid,h264_cuvid'  # 同时支持H.265和H.264
            else:
                # 默认配置
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'video_codec;h264_cuvid'
            try:
                # 直接使用RTSP URL或文件路径，让OpenCV自动处理CUDA加速
                # 显式指定使用FFMPEG后端
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    # 如果FFMPEG后端失败，尝试不指定后端
                    self.cap = cv2.VideoCapture(self.url)
                    if not self.cap.isOpened():
                        raise RuntimeError(f"Cannot open video: {url}")
            except Exception as e:
                # 如果OpenCV-CUDA失败，回退到直接使用RTSP URL
                print(f"[WARNING] OpenCV-CUDA initialization failed, falling back to regular OpenCV: {e}")
                # 回退时也显式禁用GStreamer
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.url)
                    if not self.cap.isOpened():
                        raise RuntimeError(f"Cannot open video: {url}")
        else:
            # Windows平台使用默认的OpenCV
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {url}")
        
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
        
    def add_video_stream(self, video_id, video_url, fps_limit=None, use_opencv_cuda=False):
        """
        添加视频流
        video_id: 视频唯一标识
        video_url: RTSP地址或文件路径
        fps_limit: FPS限制（可选）
        use_opencv_cuda: 是否在Jetson平台上使用OpenCV-CUDA解码（可选）
        创建buffer并传给reader
        """
        with QMutexLocker(self._mutex):
            if video_id in self._streams:
                self.log_message.emit(f"视频流 {video_id} 已存在")
                return False

            try:
                # 创建帧缓冲
                buffer = FrameBuffer(maxsize=10)
                
                # 创建并启动捕获线程
                capture_thread = threading.Thread(
                    target=self._capture_loop,
                    args=(video_id, video_url, buffer, fps_limit, use_opencv_cuda),
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

    def _capture_loop(self, video_id, video_url, buffer, fps_limit=None, use_opencv_cuda=False):
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
            reader = VideoReader(video_url, fps_limit=fps_limit, use_opencv_cuda=use_opencv_cuda)
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
                            reader = VideoReader(video_url, fps_limit=fps_limit)
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
                            reader = VideoReader(video_url, fps_limit=fps_limit)
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