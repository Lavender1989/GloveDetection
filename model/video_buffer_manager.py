# model/video_buffer_manager.py
import os
import time
from collections import deque
import cv2
import numpy as np
from datetime import datetime


class VideoBufferManager:
    """
    视频帧缓冲管理器，用于循环存储最近N秒的视频帧
    当检测到报警时，可以从缓冲区获取报警前后的视频片段
    """
    def __init__(self, buffer_seconds=10, fps=30, buffer_dir=None):
        """
        初始化视频帧缓冲管理器
        :param buffer_seconds: 缓冲区存储的秒数
        :param fps: 视频帧率（每秒帧数）
        :param buffer_dir: 临时存储目录，None表示使用默认目录
        """
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.max_frames = buffer_seconds * fps
        
        # 使用双端队列实现循环缓冲区
        self.frame_buffer = deque(maxlen=self.max_frames)
        
        # 设置临时存储目录
        if buffer_dir is None:
            self.buffer_dir = os.path.join(os.path.dirname(__file__), '..', 'temp_video_buffer')
        else:
            self.buffer_dir = buffer_dir
        
        # 确保目录存在
        if not os.path.exists(self.buffer_dir):
            os.makedirs(self.buffer_dir)
        
        # 记录最后一次清理的时间
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 300  # 5分钟清理一次临时文件
        
    def add_frame(self, frame, timestamp=None):
        """
        添加一帧到缓冲区
        :param frame: 视频帧（numpy数组）
        :param timestamp: 时间戳，如果为None则使用当前时间
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 存储帧和时间戳
        self.frame_buffer.append((timestamp, frame.copy()))
        
        # 定期清理过期的临时文件
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self._cleanup_temp_files()
            self.last_cleanup_time = current_time
    
    def get_buffer_duration(self):
        """
        获取当前缓冲区中的视频时长
        :return: 时长（秒）
        """
        if len(self.frame_buffer) < 2:
            return 0
        
        first_timestamp, _ = self.frame_buffer[0]
        last_timestamp, _ = self.frame_buffer[-1]
        return last_timestamp - first_timestamp
    
    def get_all_frames(self):
        """
        获取缓冲区中的所有帧
        :return: 帧列表，每个元素为(timestamp, frame)元组
        """
        return list(self.frame_buffer)
    
    def get_frames_since(self, timestamp):
        """
        获取指定时间戳之后的所有帧
        :param timestamp: 起始时间戳
        :return: 帧列表，每个元素为(timestamp, frame)元组
        """
        return [(ts, frame) for ts, frame in self.frame_buffer if ts >= timestamp]
    
    def save_buffer_as_video(self, output_path=None, include_timestamp=True):
        """
        将当前缓冲区中的帧保存为视频文件
        :param output_path: 输出视频路径，如果为None则自动生成
        :param include_timestamp: 是否在视频中添加时间戳
        :return: 保存的视频文件路径
        """
        if not self.frame_buffer:
            print("缓冲区为空，无法保存视频")
            return None
        
        # 确定输出路径
        if output_path is None:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.buffer_dir, f'alert_video_{timestamp_str}.mp4')
        
        # 获取第一帧以确定视频尺寸
        _, first_frame = self.frame_buffer[0]
        height, width, _ = first_frame.shape
        
        # 创建VideoWriter对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        try:
            # 写入所有帧
            for ts, frame in self.frame_buffer:
                # 如果需要，添加时间戳
                if include_timestamp:
                    frame_with_timestamp = frame.copy()
                    timestamp_text = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    # 在视频左上角添加时间戳
                    cv2.putText(frame_with_timestamp, timestamp_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    out.write(frame_with_timestamp)
                else:
                    out.write(frame)
            
            print(f"视频已保存至: {output_path}")
            return output_path
        finally:
            # 确保释放资源
            out.release()
    
    def save_buffer_as_images(self, output_dir=None, include_timestamp=True):
        """
        将当前缓冲区中的帧保存为图片序列
        :param output_dir: 输出目录，如果为None则自动生成
        :param include_timestamp: 是否在图片中添加时间戳
        :return: 保存的图片目录路径
        """
        if not self.frame_buffer:
            print("缓冲区为空，无法保存图片")
            return None
        
        # 确定输出目录
        if output_dir is None:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(self.buffer_dir, f'alert_images_{timestamp_str}')
        
        # 确保目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存所有帧
        for i, (ts, frame) in enumerate(self.frame_buffer):
            # 如果需要，添加时间戳
            if include_timestamp:
                frame_with_timestamp = frame.copy()
                timestamp_text = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                cv2.putText(frame_with_timestamp, timestamp_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                frame_to_save = frame_with_timestamp
            else:
                frame_to_save = frame
            
            # 保存为图片文件
            img_filename = f'frame_{i:05d}_{datetime.fromtimestamp(ts).strftime("%H%M%S_%f")[:-3]}.jpg'
            img_path = os.path.join(output_dir, img_filename)
            cv2.imwrite(img_path, frame_to_save)
        
        print(f"图片序列已保存至: {output_dir}")
        return output_dir
    
    def clear_buffer(self):
        """
        清空缓冲区
        """
        self.frame_buffer.clear()
        print("视频帧缓冲区已清空")
    
    def _cleanup_temp_files(self):
        """
        清理过期的临时文件（超过24小时的文件）
        """
        try:
            current_time = time.time()
            # 清理超过24小时的文件
            expire_time = current_time - 24 * 3600
            
            for filename in os.listdir(self.buffer_dir):
                file_path = os.path.join(self.buffer_dir, filename)
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < expire_time:
                    os.remove(file_path)
                    print(f"已清理过期文件: {filename}")
        except Exception as e:
            print(f"清理临时文件时发生错误: {str(e)}")
    
    def __len__(self):
        """
        返回缓冲区中的帧数
        """
        return len(self.frame_buffer)
    
    def is_buffer_full(self):
        """
        检查缓冲区是否已满
        """
        return len(self.frame_buffer) >= self.max_frames
