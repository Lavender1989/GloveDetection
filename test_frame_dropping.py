#!/usr/bin/env python3
"""
测试帧丢失问题：验证修改后的帧处理策略是否减少了帧的丢弃
"""
import os
import sys
import time
import cv2
import numpy as np
from PyQt6.QtCore import QCoreApplication, QThread
from controller.video_capture_manager import VideoCaptureManager, FrameBuffer
from controller.inference_thread import InferenceThread
from controller.types import DetectionModel

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_buffer_behavior():
    """测试帧缓冲行为"""
    print("=== 测试帧缓冲行为 ===")
    
    # 创建帧缓冲
    buffer = FrameBuffer(maxsize=10)
    
    # 模拟快速连续添加帧
    for i in range(20):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * (i % 255)  # 创建不同颜色的帧
        buffer.push(frame)
        time.sleep(0.01)  # 快速添加
        print(f"添加帧 {i}, 缓冲当前大小: {buffer.q.qsize()}")
    
    # 检查缓冲中最终保留的帧
    print(f"\n缓冲最终大小: {buffer.q.qsize()}")
    
    # 尝试获取所有帧
    frames = []
    while True:
        frame = buffer.get_latest()
        if frame is None:
            break
        frames.append(frame)
    
    print(f"从缓冲中获取到 {len(frames)} 帧")
    
    return len(frames) > 5  # 验证缓冲至少保留了5帧

def test_inference_queue_behavior():
    """测试推理队列行为"""
    print("\n=== 测试推理队列行为 ===")
    
    # 创建一个模拟的模型配置
    class MockDetectionModel:
        def __init__(self):
            self.name = "mock"
            self.enabled = True
            self.conf_threshold = 0.5
            self.device = "cpu"
            
            # 模拟模型推理
            self.model = lambda x, **kwargs: [type('MockResult', (), {'boxes': []})]
    
    models = {"mock": MockDetectionModel()}
    
    # 创建推理线程
    inference_thread = InferenceThread(models, queue_maxsize=10)
    
    # 统计成功添加的帧数
    success_count = 0
    fail_count = 0
    
    # 模拟快速连续添加帧
    for i in range(30):
        frame = np.ones((320, 320, 3), dtype=np.uint8) * (i % 255)  # 创建不同颜色的帧
        success = inference_thread.add_task(frame)
        if success:
            success_count += 1
            print(f"添加帧 {i} 成功")
        else:
            fail_count += 1
            print(f"添加帧 {i} 失败")
        
        time.sleep(0.005)  # 非常快速地添加，测试队列满时的行为
    
    print(f"\n总添加帧数: {success_count + fail_count}")
    print(f"成功添加: {success_count}")
    print(f"失败/丢弃: {fail_count}")
    
    # 停止推理线程
    inference_thread.stop()
    
    return fail_count < 5  # 验证丢弃的帧少于5帧

def test_frame_processing_rate():
    """测试帧处理速率"""
    print("\n=== 测试帧处理速率 ===")
    
    # 创建一个模拟的模型配置，模拟推理延迟
    class MockDetectionModel:
        def __init__(self):
            self.name = "mock"
            self.enabled = True
            self.conf_threshold = 0.5
            self.device = "cpu"
            
            # 模拟推理延迟（200ms）
            def mock_inference(x, **kwargs):
                time.sleep(0.2)
                return [type('MockResult', (), {'boxes': []})]
            
            self.model = mock_inference
    
    models = {"mock": MockDetectionModel()}
    
    # 创建推理线程
    inference_thread = InferenceThread(models, queue_maxsize=10)
    
    # 统计成功添加的帧数和实际处理的帧数
    success_count = 0
    processed_count = 0
    start_time = time.time()
    
    # 模拟添加帧的线程
    def add_frames():
        nonlocal success_count
        for i in range(15):  # 模拟5秒内添加15帧（3fps）
            frame = np.ones((320, 320, 3), dtype=np.uint8) * (i % 255)
            success = inference_thread.add_task(frame)
            if success:
                success_count += 1
            time.sleep(0.33)  # 大约3fps的速率添加
    
    # 监听推理完成信号
    def on_inference_done(frame, results):
        nonlocal processed_count
        processed_count += 1
        print(f"处理完成第 {processed_count} 帧")
    
    inference_thread.inference_done.connect(on_inference_done)
    
    # 启动推理线程和添加帧的线程
    inference_thread.start()
    add_thread = QThread()
    add_thread.run = add_frames
    add_thread.start()
    
    # 等待测试完成
    time.sleep(8)  # 等待足够长的时间让所有帧都被处理
    
    # 停止推理线程
    inference_thread.stop()
    add_thread.wait()
    
    elapsed_time = time.time() - start_time
    print(f"\n总耗时: {elapsed_time:.2f} 秒")
    print(f"成功添加: {success_count} 帧")
    print(f"实际处理: {processed_count} 帧")
    print(f"丢弃的帧: {success_count - processed_count}")
    print(f"处理速率: {processed_count / elapsed_time:.2f} fps")
    
    return success_count - processed_count < 3  # 验证丢弃的帧少于3帧

def main():
    """主函数"""
    print("=== 帧丢失问题测试 ===")
    
    app = QCoreApplication([])
    
    # 测试1: 帧缓冲行为
    test1_passed = test_buffer_behavior()
    print(f"\n测试1 (帧缓冲行为): {'通过' if test1_passed else '失败'}")
    
    # 测试2: 推理队列行为
    test2_passed = test_inference_queue_behavior()
    print(f"\n测试2 (推理队列行为): {'通过' if test2_passed else '失败'}")
    
    # 测试3: 帧处理速率
    test3_passed = test_frame_processing_rate()
    print(f"\n测试3 (帧处理速率): {'通过' if test3_passed else '失败'}")
    
    # 总结
    print("\n" + "="*50)
    if test1_passed and test2_passed and test3_passed:
        print("🎉 所有测试通过！帧丢失问题已得到改善。")
        return 0
    else:
        print("❌ 部分测试失败，需要进一步优化。")
        return 1

if __name__ == "__main__":
    sys.exit(main())