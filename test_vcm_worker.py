import sys
import os
import time
import cv2
import queue
from PyQt6.QtWidgets import QApplication

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from controller.video_capture_manager import VideoCaptureManager
from controller.worker import MultiDetectorWorker
from controller.grabber import FrameGrabber
from controller.scheduler import Scheduler

def test_vcm_worker_interaction():
    """测试 VideoCaptureManager 和 MultiDetectorWorker 的交互"""
    # 测试视频路径
    video_path = r"D:/detect_video/original_Video/no_wearing_gloves/20250820_1.mp4"
    
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return False
    
    print(f"=== 开始测试 VideoCaptureManager 和 MultiDetectorWorker 交互 ===")
    print(f"使用视频: {video_path}")
    
    # 1. 初始化 VideoCaptureManager
    print("\n1. 初始化 VideoCaptureManager...")
    vcm = VideoCaptureManager()
    vcm.log_message.connect(lambda msg: print(f"[VCM] {msg}"))
    
    # 2. 添加视频流
    video_id = 1
    print(f"2. 添加视频流 {video_id} -> {video_path}...")
    success = vcm.add_video_stream(
        video_id=video_id,
        video_url=video_path
    )
    
    if not success:
        print("错误: 无法添加视频流")
        return False
    
    # 3. 测试直接从 VCM 获取帧
    print("\n3. 测试直接从 VCM 获取帧...")
    time.sleep(1)
    
    for i in range(3):
        frame = vcm.get_latest_frame(video_id)
        if frame is not None:
            print(f"   直接获取帧 {i+1}: 成功，形状: {frame.shape}")
        else:
            print(f"   直接获取帧 {i+1}: 失败")
        time.sleep(0.5)
    
    # 4. 测试 FrameGrabber
    print("\n4. 测试 FrameGrabber...")
    class DummyWorker:
        def __init__(self, vcm, video_id):
            self.capture_manager = vcm
            self.video_id = video_id
        
        def get_latest(self):
            return self.capture_manager.get_latest_frame(self.video_id)
        
        def get(self, timeout=0.05):
            return self.capture_manager.get_frame(self.video_id, timeout=timeout)
    
    dummy_worker = DummyWorker(vcm, video_id)
    grabber = FrameGrabber(dummy_worker)
    
    for i in range(3):
        frame = grabber.grab()
        if frame is not None:
            print(f"   FrameGrabber 获取帧 {i+1}: 成功，形状: {frame.shape}")
        else:
            print(f"   FrameGrabber 获取帧 {i+1}: 失败")
        time.sleep(0.5)
    
    # 5. 测试 Scheduler
    print("\n5. 测试 Scheduler...")
    frame_queue = queue.Queue(maxsize=5)
    
    # 修改Scheduler的使用方式，适配新版本接口
    scheduler = Scheduler(target_fps=10)
    
    # 创建一个槽函数来处理定时触发
    def on_scheduler_tick():
        frame = grabber.grab()
        if frame is not None:
            try:
                frame_id = int(time.time() * 1000)  # 使用时间戳作为frame_id
                frame_queue.put_nowait((frame_id, frame))
            except queue.Full:
                pass
    
    # 连接tick信号到槽函数
    scheduler.tick.connect(on_scheduler_tick)
    scheduler.start()
    
    time.sleep(1)
    
    for i in range(3):
        try:
            frame_id, frame = frame_queue.get(timeout=1)
            if frame is not None:
                print(f"   Scheduler 获取帧 {i+1}: 成功，ID: {frame_id}, 形状: {frame.shape}")
        except queue.Empty:
            print(f"   Scheduler 获取帧 {i+1}: 队列为空")
    
    scheduler.stop()
    
    # 6. 测试完整的 MultiDetectorWorker
    print("\n6. 测试完整的 MultiDetectorWorker...")
    
    # 配置模型
    model_path = os.path.join(os.path.dirname(__file__), "model/glove/best.pt")
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return False
    
    models_config = {
        'glove': {
            'path': model_path,
            'target_classes': ['bare'],
            'conf': 0.7,  # 降低置信度阈值以便看到检测结果
            'threshold': 5,
            'frame_threshold': 10,
            'trigger_mode': 'area'
        }
    }
    
    # 创建一个修改过的 MultiDetectorWorker 类，添加更多调试信息
    class DebugMultiDetectorWorker(MultiDetectorWorker):
        def __init__(self, *args, **kwargs):
            # 先调用父类初始化，建立基本连接
            super().__init__(*args, **kwargs)
            self._frame_count = 0
            self._inference_count = 0
            self._result_count = 0
            
            # 确保推理线程的信号连接正确
            print(f"[DEBUG] 推理线程: {self.inference_thread}")
            print(f"[DEBUG] inference_thread 状态: 已启动={self.inference_thread.isRunning()}")
        
        def attach_video_source(self, capture_manager, video_id):
            super().attach_video_source(capture_manager, video_id)
            print(f"[DEBUG] attach_video_source 完成: capture_manager={capture_manager}, video_id={video_id}")
            print(f"[DEBUG] 内部属性: self.capture_manager={self.capture_manager}, self.video_id={self.video_id}")
        
        def on_inference_done(self, frame_np, raw_results):
            """重写处理推理结果的方法，添加更多调试信息"""
            self._result_count += 1
            print(f"\n[DEBUG] on_inference_done 被调用 (第 {self._result_count} 次)")
            print(f"[DEBUG] frame_np 形状: {frame_np.shape}")
            print(f"[DEBUG] raw_results 类型: {type(raw_results)}")
            print(f"[DEBUG] raw_results 内容: {raw_results}")
            
            # 直接打印原始推理结果中的 boxes 信息
            for model_name, result in raw_results.items():
                if result is not None:
                    try:
                        print(f"\n[DEBUG] 模型 {model_name} 的推理结果:")
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            print(f"[DEBUG]   Boxes: {result.boxes}")
                            if hasattr(result.boxes, 'xyxy'):
                                print(f"[DEBUG]   Boxes xyxy: {result.boxes.xyxy.cpu().numpy()}")
                            if hasattr(result.boxes, 'conf'):
                                print(f"[DEBUG]   Boxes conf: {result.boxes.conf.cpu().numpy()}")
                            if hasattr(result.boxes, 'cls'):
                                print(f"[DEBUG]   Boxes cls: {result.boxes.cls.cpu().numpy()}")
                            if hasattr(result, 'names'):
                                print(f"[DEBUG]   类别名称: {result.names}")
                            
                            # 直接处理原始结果
                            boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, 'xyxy') else []
                            confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else []
                            cls_ids = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, 'cls') else []
                            
                            detected_objects = []
                            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                                cls_name = result.names[int(cls_id)] if hasattr(result, 'names') else f'class_{int(cls_id)}'
                                detected_objects.append(f"{cls_name} (conf: {conf:.2f})")
                            
                            print(f"[DEBUG]   检测到的目标: {detected_objects}")
                        else:
                            print(f"[DEBUG]   无检测框")
                    except Exception as e:
                        print(f"[DEBUG]   处理推理结果错误: {e}")
                        import traceback
                        traceback.print_exc()
            
            # 调用父类的 on_inference_done 方法继续处理
            print("[DEBUG] 调用父类 on_inference_done 方法")
            super().on_inference_done(frame_np, raw_results)
    
    worker = DebugMultiDetectorWorker(
        models_config=models_config,
        video_name="Test Video",
        view_index=0,
        alert_email="",
        capture_manager=vcm,
        video_id=video_id
    )
    
    # 连接信号
    worker.log_message.connect(lambda msg: print(f"[WORKER] {msg}"))
    worker.alert_message.connect(lambda msg: print(f"[ALERT] {msg}"))
    worker.proc_frame_ready.connect(lambda qimg: print(f"[FRAME] 收到处理后帧: {qimg.width()}x{qimg.height()}"))
    worker.detection_result.connect(lambda det_type, results: 
        print(f"[DETECTION] {det_type} 检测到 {len(results)} 个目标: {results}") if results else 
        print(f"[DETECTION] {det_type} 未检测到目标")
    )
    
    # 附加视频源
    print("   附加视频源...")
    worker.attach_video_source(vcm, video_id)
    
    # 测试 grabber 是否能正常工作
    print("   测试 worker.grabber...")
    for i in range(3):
        frame = worker.grabber.grab()
        if frame is not None:
            print(f"   worker.grabber 获取帧 {i+1}: 成功")
            # 显示一小部分帧数据以确认
            print(f"      帧数据预览: {frame[0, 0, :]}")
        else:
            print(f"   worker.grabber 获取帧 {i+1}: 失败")
        time.sleep(0.5)
    
    # 启动 worker
    print("   启动 worker...")
    worker.start()
    
    # 使用Qt的事件循环来处理信号
    print("   启动Qt事件循环...")
    
    # 设置定时器，在指定时间后停止测试
    def stop_test():
        print(f"\n[STATS] 总处理帧数: {worker._frame_count}")
        print(f"[STATS] 总推理任务数: {worker._inference_count}")
        print(f"[STATS] 总推理结果数: {worker._result_count}")
        
        # 停止所有组件
        print("\n7. 停止所有组件...")
        worker.stop()
        vcm.stop()
        
        print("\n=== 测试完成 ===")
        return True
    
    return stop_test

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    try:
        # 创建一个定时器，确保测试能够在一段时间后自动停止
        from PyQt6.QtCore import QTimer
        
        # 获取测试结果
        stop_test = test_vcm_worker_interaction()
        
        # 设置定时器，在指定时间后停止测试
        def stop_app():
            stop_test()
            app.quit()
        
        # 5秒后停止应用
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(stop_app)
        timer.start(5000)  # 5秒后停止
        
        # 启动事件循环
        app.exec()
        
    except Exception as e:
        print(f"测试过程中发生错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 退出应用
    sys.exit(0)