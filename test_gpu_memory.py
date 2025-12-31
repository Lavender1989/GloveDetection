import torch
import sys
import os
from controller.types import DetectionModel, ModelManager

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_sharing():
    """测试模型共享机制是否正常工作"""
    print("=== 测试模型共享机制 ===")
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA不可用，无法测试GPU内存使用情况")
        return
    
    # 打印初始GPU内存使用情况
    print(f"初始GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # 模型路径
    glove_model_path = os.path.join(os.path.dirname(__file__), "model", "glove", "best.pt")
    head_model_path = os.path.join(os.path.dirname(__file__), "model", "head", "best.pt")
    
    # 创建多个DetectionModel实例，测试模型共享
    print("\n创建多个DetectionModel实例...")
    
    # 记录创建时间和内存使用
    import time
    start_time = time.time()
    
    models = []
    for i in range(5):  # 创建5个模型实例
        model1 = DetectionModel(
            name=f"glove_{i}",
            model_path=glove_model_path,
            target_classes=['bare'],
            conf_threshold=0.7
        )
        model2 = DetectionModel(
            name=f"head_{i}",
            model_path=head_model_path,
            target_classes=['touch'],
            conf_threshold=0.7
        )
        models.append((model1, model2))
        
        # 打印当前内存使用
        print(f"创建第{i+1}组模型后，GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    end_time = time.time()
    print(f"\n创建5组模型耗时: {end_time - start_time:.2f}秒")
    print(f"最终GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # 验证模型是否共享
    print("\n验证模型共享...")
    if len(models) > 1:
        # 检查所有glove模型是否指向同一个实例
        glove_models = [model[0].model for model in models]
        is_glove_shared = all(model is glove_models[0] for model in glove_models)
        print(f"Glove模型是否共享: {'是' if is_glove_shared else '否'}")
        
        # 检查所有head模型是否指向同一个实例
        head_models = [model[1].model for model in models]
        is_head_shared = all(model is head_models[0] for model in head_models)
        print(f"Head模型是否共享: {'是' if is_head_shared else '否'}")
    
    # 释放资源
    print("\n释放所有模型...")
    model_manager = ModelManager()
    model_manager.release_all_models()
    
    print(f"释放后GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print("=== 测试完成 ===")

if __name__ == "__main__":
    test_model_sharing()
