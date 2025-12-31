import cv2
import sys
import time
import numpy as np
from typing import Dict, List

# 添加项目根目录到Python路径
sys.path.append('d:\GloveDetection')

from controller.inference_thread import InferenceThread
from controller.postprocess import PostProcessor
from controller.types import DetectionModel


def test_model_inference():
    """
    测试模型推理和后处理流程，重点关注类别名称
    """
    print("=== 模型推理测试开始 ===")
    
    # 1. 配置参数
    video_path = r"D:\detect_video\original_Video\no_wearing_gloves\0911_1.mp4"
    model_path = r"model\glove\best.pt"
    
    # 2. 初始化检测模型
    print("\n初始化检测模型...")
    models_config = {
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
    
    # 3. 初始化推理线程
    print("初始化推理线程...")
    inference_thread = InferenceThread(models_config)
    inference_thread.start()
    time.sleep(1)  # 等待线程启动
    
    # 4. 初始化后处理器
    print("初始化后处理器...")
    postprocessor = PostProcessor(models_config)
    
    # 5. 打开视频文件
    print(f"打开视频文件: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频信息: 总帧数={total_frames}, FPS={fps}")
    
    # 6. 逐帧处理
    frame_count = 0
    processing_times = []
    max_test_frames = 10  # 只测试前10帧
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_test_frames:
            break
        
        frame_count += 1
        print(f"\n=== 处理第 {frame_count}/{total_frames} 帧 ===")
        
        # 记录处理时间
        start_time = time.time()
        
        # 7. 直接使用模型进行推理
        try:
            # 使用模型进行推理
            print("开始模型推理...")
            results = models_config['glove'].model(
                frame,
                conf=0.8,
                device='cuda',
                imgsz=320,
                half=True,
                verbose=False
            )[0]
            
            # 8. 检查原始推理结果
            print("\n=== 原始推理结果分析 ===")
            print(f"检测到目标数量: {len(results.boxes)}")
            
            # 查看模型的类别列表
            model_names = models_config['glove'].model.names
            print(f"模型类别列表: {model_names}")
            
            # 解析原始结果的每个检测
            if len(results.boxes) > 0:
                for i, (box, conf, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
                    cls_idx = int(cls.item())
                    cls_name = model_names[cls_idx] if cls_idx < len(model_names) else f"未知类别({cls_idx})"
                    
                    print(f"\n原始检测 {i+1}:")
                    print(f"  类别索引: {cls_idx}")
                    print(f"  类别名称: {cls_name}")
                    print(f"  置信度: {conf.item():.4f}")
                    print(f"  边界框: {box.tolist()}")
            
            # 9. 使用后处理器解析结果
            print("\n=== 后处理器解析结果 ===")
            raw_results = {'glove': results}
            structured_results = postprocessor.parse_raw_results(raw_results)
            
            for model_name, results in structured_results.items():
                print(f"{model_name}模型解析后检测到 {len(results)} 个目标:")
                for i, result in enumerate(results):
                    print(f"  检测 {i+1}:")
                    print(f"    - 模型名: {result.model_name}")
                    print(f"    - 类别名: {result.class_name}")
                    print(f"    - 置信度: {result.confidence:.4f}")
                    print(f"    - 边界框: {result.bbox}")
            
            # 10. 后处理
            print("\n=== 后处理分析 ===")
            need_alert, info = postprocessor.update_state_and_check_alert(structured_results)
            print(f"需要告警: {need_alert}")
            print(f"危险检测: {info['danger_detected']}")
            print(f"危险框: {info['danger_boxes']}")
            print(f"状态历史: {info['history']}")
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            import traceback
            traceback.print_exc()
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        print(f"\n帧处理时间: {processing_time:.4f}秒")
    
    # 11. 统计信息
    cap.release()
    inference_thread.stop()
    
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"\n=== 测试完成 ===")
        print(f"总处理帧数: {frame_count}")
        print(f"平均处理时间: {avg_time:.4f}秒")
        print(f"平均FPS: {1/avg_time:.2f}")
    
    print("=== 模型推理测试结束 ===")


if __name__ == "__main__":
    test_model_inference()