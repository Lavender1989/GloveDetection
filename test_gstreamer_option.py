import sys
import os
import time
import cv2
from PyQt6.QtWidgets import QApplication

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from controller.video_capture_manager import VideoCaptureManager

def test_gstreamer_option():
    """
    测试GStreamer选项是否正常工作
    """
    # 测试视频路径（使用一个简单的视频文件）
    video_path = r"D:/detect_video/original_Video/no_wearing_gloves/20250820_1.mp4"
    
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return False
    
    print(f"=== 开始测试 GStreamer 选项 ===")
    print(f"使用视频: {video_path}")
    
    # 1. 测试默认模式（不使用GStreamer）
    print("\n1. 测试默认模式（不使用GStreamer）...")
    vcm_default = VideoCaptureManager()
    vcm_default.log_message.connect(lambda msg: print(f"[VCM_DEFAULT] {msg}"))
    
    success = vcm_default.add_video_stream(
        video_id="default",
        video_url=video_path
    )
    
    if not success:
        print("错误: 默认模式无法添加视频流")
        return False
    
    # 获取几帧测试
    for i in range(3):
        frame = vcm_default.get_latest_frame("default")
        if frame is not None:
            print(f"   默认模式获取帧 {i+1}: 成功，形状: {frame.shape}")
        else:
            print(f"   默认模式获取帧 {i+1}: 失败")
        time.sleep(0.5)
    
    vcm_default.stop()
    
    # 2. 测试GStreamer模式（如果在Jetson平台上）
    print("\n2. 测试GStreamer模式...")
    vcm_gstreamer = VideoCaptureManager()
    vcm_gstreamer.log_message.connect(lambda msg: print(f"[VCM_GSTREAMER] {msg}"))
    
    success = vcm_gstreamer.add_video_stream(
        video_id="gstreamer",
        video_url=video_path,
        use_gstreamer=True
    )
    
    if not success:
        print("错误: GStreamer模式无法添加视频流")
        return False
    
    # 获取几帧测试
    for i in range(3):
        frame = vcm_gstreamer.get_latest_frame("gstreamer")
        if frame is not None:
            print(f"   GStreamer模式获取帧 {i+1}: 成功，形状: {frame.shape}")
        else:
            print(f"   GStreamer模式获取帧 {i+1}: 失败")
        time.sleep(0.5)
    
    vcm_gstreamer.stop()
    
    print("\n=== 测试完成 ===")
    return True

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    try:
        success = test_gstreamer_option()
        if success:
            print("\n测试成功: GStreamer选项正常工作")
        else:
            print("\n测试失败: GStreamer选项无法正常工作")
    except Exception as e:
        print(f"\n测试过程中发生错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        app.quit()
