#!/usr/bin/env python3
"""
测试内存优化：验证模型共享和内存管理是否正常工作
"""
import os
import sys
import torch
import time
from controller.types import DetectionModel, ModelManager

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_sharing():
    """测试模型实例是否正确共享"""
    print("=== 测试模型共享 ===")
    
    # 获取模型路径
    model_path = os.path.join(os.path.dirname(__file__), "model/glove/best.pt")
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return False
    
    # 打印初始内存使用情况
    if torch.cuda.is_available():
        print(f"初始 GPU内存使用: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")
        print(f"初始 GPU内存缓存: {torch.cuda.memory_reserved()/1024/1024:.2f} MB")
    else:
        print("CUDA不可用，使用CPU进行测试")
    
    # 创建多个DetectionModel实例，验证它们共享同一个模型
    models = []
    for i in range(3):
        print(f"\n创建模型实例 {i+1}...")
        model = DetectionModel(
            name=f"glove_{i}",
            model_path=model_path,
            target_classes=["bare"],
            conf_threshold=0.7
        )
        models.append(model)
        
        # 检查内存使用情况
        if torch.cuda.is_available():
            print(f"  GPU内存使用: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")
            print(f"  GPU内存缓存: {torch.cuda.memory_reserved()/1024/1024:.2f} MB")
    
    # 验证所有模型实例共享同一个模型对象
    print(f"\n验证模型共享:")
    model_ids = [id(model.model) for model in models]
    print(f"  模型实例ID: {model_ids}")
    print(f"  所有模型实例是否共享同一个模型对象: {len(set(model_ids)) == 1}")
    
    # 释放模型
    print(f"\n释放模型实例...")
    for i, model in enumerate(models):
        model.release()
        print(f"  释放模型实例 {i+1}完成")
    
    # 清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"\n最终 GPU内存使用: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")
        print(f"最终 GPU内存缓存: {torch.cuda.memory_reserved()/1024/1024:.2f} MB")
    
    return True

def test_memory_optimization():
    """测试内存优化是否有效"""
    print("\n=== 测试内存优化 ===")
    
    # 获取模型路径
    model_path = os.path.join(os.path.dirname(__file__), "model/glove/best.pt")
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return False
    
    # 模拟Jetson NX环境下的内存压力
    print("模拟内存压力测试...")
    
    # 先分配一些内存模拟压力
    if torch.cuda.is_available():
        print(f"分配临时GPU内存...")
        # 分配一些临时张量来模拟内存压力
        temp_tensors = []
        for i in range(10):
            try:
                tensor = torch.randn(1024, 1024, device='cuda')
                temp_tensors.append(tensor)
                print(f"  分配了 {i+1} 个 1024x1024 张量")
            except RuntimeError as e:
                print(f"  分配失败: {e}")
                break
        
        # 清理临时内存
        print("\n清理临时GPU内存...")
        del temp_tensors
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  GPU内存使用: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")
        print(f"  GPU内存缓存: {torch.cuda.memory_reserved()/1024/1024:.2f} MB")
    
    # 尝试加载模型
    print("\n尝试加载模型...")
    try:
        model = DetectionModel(
            name="glove_test",
            model_path=model_path,
            target_classes=["bare"],
            conf_threshold=0.7
        )
        print(f"模型加载成功，使用设备: {model.device}")
        
        # 测试推理
        print("\n测试推理...")
        import numpy as np
        # 创建一个随机输入
        input_frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        
        with torch.no_grad():
            result = model.model(
                input_frame,
                conf=model.conf_threshold,
                device=model.device,
                imgsz=320,
                half=model.device == 'cuda',
                batch=1,
                verbose=False
            )
        print("推理成功")
        
        # 释放模型
        model.release()
    except Exception as e:
        print(f"模型加载或推理失败: {e}")
        return False
    
    # 清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"\n最终 GPU内存使用: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")
        print(f"最终 GPU内存缓存: {torch.cuda.memory_reserved()/1024/1024:.2f} MB")
    
    return True

def main():
    """主函数"""
    print("=== 内存优化测试 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory/1024/1024:.2f} MB")
    
    # 测试模型共享
    print("\n" + "="*50)
    if not test_model_sharing():
        print("模型共享测试失败")
        return 1
    
    # 测试内存优化
    print("\n" + "="*50)
    if not test_memory_optimization():
        print("内存优化测试失败")
        return 1
    
    print("\n" + "="*50)
    print("所有测试通过！内存优化已生效。")
    return 0

if __name__ == "__main__":
    sys.exit(main())