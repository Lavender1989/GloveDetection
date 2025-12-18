# -*- coding: utf-8 -*-
"""
跨平台视频捕获管理器
- Windows平台：使用OpenCV CPU解码
- Jetson平台：使用GStreamer + NVDEC硬件解码
- 提供统一的torch tensor接口
- 支持多路RTSP流管理
- 可配置缓冲策略
"""

import platform
import time
import threading
import queue
import cv2
from time import sleep
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker

# Windows 兼容依赖
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GObject
    Gst.init(None)
    _GSTREAMER_AVAILABLE = True
except (ImportError, ValueError):
    _GSTREAMER_AVAILABLE = False

# =============== 1. 自动选择解码后端 ===============
class VideoBackendSelector:
    @staticmethod
    def is_jetson():
        # Jetson 会显示为 aarch64
        return platform.machine() == "aarch64"
    
    @staticmethod
    def build_rtsp_pipeline(url: str):
        return (
            f"rtspsrc location={url} latency=500 protocols=udp ! "
            "queue max-size-buffers=1 ! "
            "rtph264depay ! h264parse ! "
            "nvv4l2decoder enable-max-performance=1 ! "
            "nvvidconv ! "
            "video/x-raw,format=BGRx ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=appsink drop=true sync=false max-buffers=1"
        )

    @staticmethod
    def build_file_pipeline(path: str):
        return (
            f"filesrc location={path} ! "
            "qtdemux ! h264parse ! "
            "nvv4l2decoder enable-max-performance=1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=true sync=false"
        )


# =============== 2. 跨平台 VideoReader ===============
class VideoReader:
    def __init__(self, url, fps_limit=None):
        """
        跨平台 VideoReader
        Windows: OpenCV
        Jetson: GStreamer + NVDEC
        """
        self.url = url
        self.is_file = not url.lower().startswith("rtsp://")  # 判断是不是rtsp视频
        self._use_gst = VideoBackendSelector.is_jetson() and _GSTREAMER_AVAILABLE
        self.fps_limit = fps_limit
        self._frame = None
        self._running = True

        if self._use_gst:
            self._build_gst_pipeline()
        else:
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {url}")
            # 获取视频 FPS（无论 RTSP 或文件）
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps_dt = 1.0 / (fps if fps > 1 or fps <= 120 else 30)

       
     # ================= Jetson GStreamer 部分 =================
    def _build_gst_pipeline(self):
        from gi.repository import Gst
        if self.is_file:
            pipeline_str = VideoBackendSelector.build_file_pipeline(self.url)
        else:
            pipeline_str = VideoBackendSelector.build_rtsp_pipeline(self.url)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("appsink")
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", True)
        self.appsink.connect("new-sample", self._on_new_sample)
        self.pipeline.set_state(Gst.State.PLAYING)

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        import numpy as np
        result, mapinfo = buf.map(Gst.MapFlags.READ)
        if result:
            h = caps.get_structure(0).get_value('height')
            w = caps.get_structure(0).get_value('width')
            format = struct.get_string("format")  # 如BGRx/BGRA
            channels = 4 if "BGRx" in format or "BGRA" in format else 3
            frame_size = mapinfo.size
            expected_size = h * w * channels
            if frame_size != expected_size:
                print(f"[WARN] 帧尺寸不匹配：实际{frame_size} ≠ 预期{h}×{w}×{channels}={expected_size}")
                # 容错：按实际尺寸reshape（避免崩溃）
                frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
                # 尝试自动推导尺寸（兜底）
                frame = frame.reshape((h, w, -1))
            else:
                frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, channels))
            
            # 5. 转换为3通道BGR（如果是4通道）
            if channels == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # 后续处理帧...
            return Gst.FlowReturn.OK

    def read(self):
        if self._use_gst:
            ret = self._frame is not None
            frame = self._frame
            if ret:
                print(f"[DEBUG] VideoReader (GST) 成功读取帧: {frame.shape}")
            else:
                print(f"[DEBUG] VideoReader (GST) 未读取到帧")
            return ret, frame
        else:
            ret, frame = self.cap.read()
            if ret:
                print(f"[DEBUG] VideoReader (OpenCV) 成功读取帧: {frame.shape}")
            else:
                print(f"[DEBUG] VideoReader (OpenCV) 未读取到帧")
            if self.fps_limit:
                time.sleep(1.0 / self.fps_limit)
            return ret, frame

    def release(self):
        self._running = False
        if self._use_gst:
            self.pipeline.set_state(Gst.State.NULL)
        else:
            self.cap.release()


# =============== 3. 线程安全 Buffer ===============
class FrameBuffer:
    # 单个视频流的帧缓冲
    def __init__(self, maxsize=5):
        self.q = queue.Queue(maxsize=maxsize)

    def push(self, frame):
        print(f"[DEBUG] FrameBuffer.push: 收到帧，形状: {frame.shape}")
        if self.q.full():
            try:
                old_frame = self.q.get_nowait()  # 丢掉最旧帧
                print(f"[DEBUG] FrameBuffer.push: 队列已满，丢掉旧帧")
            except queue.Empty:
                pass
        self.q.put_nowait(frame)
        print(f"[DEBUG] FrameBuffer.push: 帧已入队，当前队列大小: {self.q.qsize()}")

    def get_latest(self):
        # LIFO 策略：获取最新帧
        frame = None
        count = 0
        while not self.q.empty():
            frame = self.q.get()
            count += 1
        if frame is not None:
            print(f"[DEBUG] FrameBuffer.get_latest: 获取到最新帧，形状: {frame.shape}，共处理 {count} 帧")
        else:
            print(f"[DEBUG] FrameBuffer.get_latest: 未获取到帧")
        return frame

# =============== 4. VideoCaptureManager（多视频流管理器） ===============
class VideoCaptureManager(QObject):
    """
    多视频流管理器：管理多个视频源的捕获线程和缓冲
    为每个视频源创建独立的捕获线程和帧缓冲
    """
    log_message = pyqtSignal(str)
    rtsp_disconnected = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._streams = {}  # {video_id: (capture_thread, buffer)}
        self._mutex = QMutex()
        self._running = True
        self._paused_streams = set()  # 暂停中的视频流ID

    def add_video_stream(self, video_id, video_url, fps_limit=None):
        """
        添加视频流
        video_id: 视频唯一标识
        video_url: RTSP地址或文件路径
        force_gstreamer: 是否强制使用GStreamer（可选）
        fps_limit: FPS限制（可选）
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
                    args=(video_id, video_url, buffer, fps_limit),
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

    def _capture_loop(self, video_id, video_url, buffer, fps_limit=None):
        """
        单个视频流的捕获循环
        """
        try:
            print(f"[DEBUG] VideoCaptureManager._capture_loop: 开始捕获视频流 {video_id} - {video_url}")
            # 创建视频读取器
            reader = VideoReader(video_url, fps_limit=fps_limit)
            fps_dt = 1.0 / fps_limit if fps_limit else getattr(reader, 'fps_dt', 1/30)
            
            last_t = 0
            reconnect_count = 0
            was_reconnecting = False # 是否需要重连
            frame_count = 0
            success_count = 0

            while self._running:
                # ⭐ 暂停检查
                if video_id in self._paused_streams:
                    print(f"[DEBUG] VideoCaptureManager._capture_loop: 视频流 {video_id} 已暂停")
                    time.sleep(0.02)
                    continue
                # 按FPS限速
                if fps_dt:
                    now = time.time()
                    if now - last_t < fps_dt:
                        time.sleep(0.001)
                        continue
                    last_t = now
                
                frame_count += 1
                print(f"[DEBUG] VideoCaptureManager._capture_loop: 第 {frame_count} 次尝试读取视频流 {video_id}")
                ret, frame = reader.read()
                if not ret or frame is None:
                    print(f"[DEBUG] VideoCaptureManager._capture_loop: 视频流 {video_id} 读取失败")
                    if reader.is_file:
                        # ⭐ 本地视频：播放完毕，需要循环播放
                        self.log_message.emit(f"[视频 {video_id}] 播放完毕，重新开始播放")
                        print(f"[DEBUG] VideoCaptureManager._capture_loop: 本地视频 {video_id} 播放完毕，重新开始")
                        try:
                            reader.release()
                            reader = VideoReader(video_url, fps_limit=fps_limit)
                            last_t = 0
                            continue
                        except Exception as e:
                            self.log_message.emit(f"[视频 {video_id}] 重新打开视频文件失败: {e}")
                            print(f"[DEBUG] VideoCaptureManager._capture_loop: 重新打开视频文件 {video_id} 失败: {e}")
                            time.sleep(1.0)
                            continue

                    else:
                        reconnect_count += 1
                        was_reconnecting = True  # 正在进行重连
                        self.log_message.emit(f"[视频 {video_id}] 连接丢失，正在重连... ({reconnect_count})")
                        print(f"[DEBUG] VideoCaptureManager._capture_loop: 视频流 {video_id} 连接丢失，正在重连... ({reconnect_count})")
                        time.sleep(1.0)
                    
                        # 重连机制
                        if reconnect_count > 5:
                            self.log_message.emit(f"[视频 {video_id}] 重连失败，放弃")
                            print(f"[DEBUG] VideoCaptureManager._capture_loop: 视频流 {video_id} 重连失败，放弃")
                            self.rtsp_disconnected.emit(video_id)
                            break
                        try:
                            reader.release()
                            reader = VideoReader(video_url, fps_limit=fps_limit)
                            reconnect_count = 0
                            print(f"[DEBUG] VideoCaptureManager._capture_loop: 视频流 {video_id} 重连成功")
                        except Exception as e:
                            self.log_message.emit(f"[视频 {video_id}] 重连失败: {e}")
                            print(f"[DEBUG] VideoCaptureManager._capture_loop: 视频流 {video_id} 重连失败: {e}")
                            time.sleep(1.0)
                    continue
                
                success_count += 1
                print(f"[DEBUG] VideoCaptureManager._capture_loop: 视频流 {video_id} 读取成功，帧形状: {frame.shape}，成功次数: {success_count}/{frame_count}")
                
                # 过滤全黑帧
                if not frame.any():
                    print(f"[DEBUG] VideoCaptureManager._capture_loop: 视频流 {video_id} 读取到全黑帧，跳过")
                    continue

                if was_reconnecting:
                    self.log_message.emit(f"[视频 {video_id}] 重连成功")
                    print(f"[DEBUG] VideoCaptureManager._capture_loop: 视频流 {video_id} 重连成功")
                    was_reconnecting = False  # 重连成功，重置标志
                    reconnect_count = 0
                
                print(f"[DEBUG] VideoCaptureManager._capture_loop: 视频流 {video_id} 将帧推入缓冲区")
                buffer.push(frame)
                
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
    video = VideoCaptureManager(r"D:\detect_video\original_Video\no_wearing_gloves\0911_1.mp4", buffer)
    video.start()

    print("Reader started. Press Ctrl+C to stop.")

    while True:
        frame = buffer.get()
        if frame is not None:
            cv2.imshow("Test Frame", frame)

        # 必须加，否则窗口不刷新
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    sleep(0.01)
    video.stop()
    cv2.destroyAllWindows()