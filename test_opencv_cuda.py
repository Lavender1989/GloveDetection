#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试OpenCV CUDA视频读取功能
"""

import sys
import os
import cv2
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from controller.video_capture_manager import VideoReader, VideoBackendSelector

def test_video_reader(url, use_opencv_cuda=False, use_gstreamer=False):
    """
    测试VideoReader类
    
    参数：
    - url: 视频路径或RTSP地址
    - use_opencv_cuda: 是否使用OpenCV CUDA加速
    - use_gstreamer: 是否使用GStreamer硬件解码
    """
    print(f"\n[测试] 视频URL: {url}")
    print(f"[测试] 使用OpenCV CUDA: {use_opencv_cuda}")
    print(f"[测试] 使用GStreamer: {use_gstreamer}")
    
    try:
        # 创建VideoReader实例
        reader = VideoReader(url, use_opencv_cuda=use_opencv_cuda, use_gstreamer=use_gstreamer)
        
        # 测试视频读取
        start_time = time.time()
        frame_count = 0
        max_frames = 100
        
        while frame_count < max_frames:
            ret, frame = reader.read()
            if not ret or frame is None:
                print(f"[测试] 读取帧失败，已读取 {frame_count} 帧")
                break
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"[测试] 已读取 {frame_count} 帧")
        
        end_time = time.time()
        fps = frame_count / (end_time - start_time + 1e-9)
        
        print(f"[测试] 测试完成，共读取 {frame_count} 帧")
        print(f"[测试] 平均FPS: {fps:.2f}")
        
        # 释放资源
        reader.release()
        return True
        
    except Exception as e:
        print(f"[测试] 出错: {e}")
        return False

def main():
    """
    主测试函数
    """
    # 检查是否支持CUDA
    print(f"CUDA可用: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"CUDA设备数量: {cv2.cuda.getCudaEnabledDeviceCount()}")
    
    # 检查是否是Jetson平台
    print(f"Jetson平台: {VideoBackendSelector.is_jetson()}")
    
    # 获取测试视频路径
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    else:
        video_url = "rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/501"
    
    if not video_url:
        print("[错误] 未提供视频路径")
        return 1
    
    # 测试不同配置
    print("\n=== 测试1: 使用默认配置 ===")
    test_video_reader(video_url)
    
    print("\n=== 测试2: 使用OpenCV CUDA ===")
    test_video_reader(video_url, use_opencv_cuda=True)
    
    print("\n=== 测试3: 使用GStreamer ===")
    test_video_reader(video_url, use_gstreamer=True)
    
    print("\n=== 测试4: 同时使用OpenCV CUDA和GStreamer ===")
    test_video_reader(video_url, use_opencv_cuda=True, use_gstreamer=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

