import cv2
import sys
import time
import numpy as np
from typing import Dict, List
from PyQt6.QtCore import QCoreApplication, QObject, pyqtSlot, QTimer

# 添加项目根目录到Python路径
sys.path.append('d:\GloveDetection')

from controller.inference_thread import InferenceThread
from controller.postprocess import PostProcessor
from controller.types import DetectionModel
from controller.drawer import Drawer
from controller.region_loader import load_area_for_view

class TestRunner(QObject):
    """
    使用PyQt信号槽机制的测试运行器
    """
    def __init__(self, video_path, model_path):
        super().__init__()
        
        self.video_path = video_path
        self.model_path = model_path
        self.cap = None
        self.frame_count = 0
        self.total_frames = 0
        self.processing_times = []
        
        # 初始化检测模型
        print("\n初始化检测模型...")
        self.models_config = {
            'glove': DetectionModel(
                name='glove',
                model_path=model_path,
                target_classes=['bare', 'wear'],  # 包含新训练的两个类别
                conf_threshold=0.8,
                frame_threshold=2,
                trigger_mode='area',
                enabled=True
            )
        }
        
        # 初始化推理线程
        print("初始化推理线程...")
        self.inference_thread = InferenceThread(self.models_config)
        self.inference_thread.inference_done.connect(self.on_inference_done)
        self.inference_thread.inference_error.connect(self.on_inference_error)
        self.inference_thread.start()
        time.sleep(1)  # 等待线程启动
        
        # 初始化后处理器
        print("初始化后处理器...")
        self.postprocessor = PostProcessor(self.models_config)
        
        # 初始化绘制器
        print("初始化绘制器...")
        self.drawer = Drawer()
        
        # 初始化区域加载器
        self.view_index = 0  # 默认使用第一个视图
        
        # 初始化定时器用于控制处理速度
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)
        
    def start(self):
        """开始测试"""
        print(f"\n打开视频文件: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"错误: 无法打开视频文件 {self.video_path}")
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # 加载区域
        print("\n加载检测区域...")
        self.area_boxes, area_log = load_area_for_view(self.view_index, self.frame_width, self.frame_height)
        print(f"区域加载结果: {area_log}")
        print(f"有效区域数量: {len(self.area_boxes)}")
        
        print(f"视频信息: 总帧数={self.total_frames}, FPS={fps}, 分辨率={self.frame_width}x{self.frame_height}")
        
        # 设置定时器以视频FPS的速度处理帧
        self.timer.start(int(1000 / fps))
        return True
    
    @pyqtSlot()
    def process_next_frame(self):
        """处理下一帧"""
        ret, frame = self.cap.read()
        if not ret:
            self.finish_test()
            return
        
        self.frame_count += 1
        print(f"\n处理第 {self.frame_count}/{self.total_frames} 帧")
        
        # 记录开始时间
        self.current_frame = frame
        self.current_frame_time = time.time()
        
        # 添加帧到推理队列
        self.inference_thread.add_task(frame)
    
    @pyqtSlot(object, object)
    def on_inference_done(self, frame_np, raw_results):
        """处理推理完成信号"""
        processing_time = time.time() - self.current_frame_time
        self.processing_times.append(processing_time)
        
        print(f"推理完成，处理时间: {processing_time:.4f}秒")
        
        # 解析原始结果
        structured_results = self.postprocessor.parse_raw_results(raw_results)
        
        # 打印检测结果
        print(f"检测结果:")
        for model_name, results in structured_results.items():
            print(f"  {model_name}模型检测到 {len(results)} 个目标:")
            for result in results:
                print(f"    - 类别: {result.class_name}, 置信度: {result.confidence:.4f}, 框: {result.bbox}")
        
        # 后处理
        need_alert, info = self.postprocessor.update_state_and_check_alert(structured_results)
        print(f"后处理结果: need_alert={need_alert}")
        print(f"危险检测: {info['danger_detected']}")
        
        # 绘制检测结果
        annotated_frame = self.drawer.draw(
            frame_np,
            structured_results,
            info['danger_detected'],
            info['danger_boxes'],
            self.area_boxes,
            need_alert,
            0.0  # 模拟报警剩余时间
        )
        
        # 显示当前帧
        cv2.imshow('Test Frame with Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.finish_test()
    
    @pyqtSlot(str)
    def on_inference_error(self, error_msg):
        """处理推理错误信号"""
        print(f"推理错误: {error_msg}")
    
    def finish_test(self):
        """完成测试"""
        print(f"\n=== 测试完成 ===")
        
        # 清理资源
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.inference_thread.stop()
        cv2.destroyAllWindows()
        
        # 打印统计信息
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            print(f"总处理帧数: {self.frame_count}")
            print(f"平均处理时间: {avg_time:.4f}秒")
            print(f"平均FPS: {1/avg_time:.2f}")
        
        print("=== 测试结束 ===")
        
        # 退出应用
        QCoreApplication.quit()

def main():
    """主函数"""
    print("=== 模型推理测试程序 ===")
    print("此程序用于测试新训练的模型是否能正常运行")
    print("支持bare和wear两个类别")
    print()
    
    # 获取用户输入
    video_path = input("请输入测试视频文件路径: ")
    model_path = input("请输入新训练的模型路径: ")
    
    # 创建应用实例
    app = QCoreApplication(sys.argv)
    
    # 创建测试运行器
    runner = TestRunner(video_path, model_path)
    
    # 开始测试
    if runner.start():
        # 运行应用事件循环
        sys.exit(app.exec())
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()