import torch
import sys
import os
import threading
import time
import numpy as np

# 首先打印基本信息
try:
    print(f"Python路径: {sys.executable}")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本路径: {os.path.abspath(__file__)}")
    
    # 检查CUDA
    print(f"torch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
        print(f"GPU内存已用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    print(f"项目根目录: {project_root}")
    
    # 尝试导入模型
    print("\n尝试导入DetectionModel...")
    from controller.types import DetectionModel
    print("✅ 成功导入DetectionModel")
    
    # 测试模型路径
    model_path = os.path.join(project_root, "model", "glove", "best.pt")
    print(f"模型路径: {model_path}")
    print(f"模型文件存在: {os.path.exists(model_path)}")
    
    # 加载模型
    print("\n正在加载模型...")
    shared_model = DetectionModel(
        name="shared_glove_model",
        model_path=model_path,
        target_classes=['bare'],
        conf_threshold=0.7
    )
    print("✅ 模型加载成功")
    print(f"模型设备: {shared_model.device}")
    print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # 创建测试数据
    print("\n创建测试数据...")
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"测试图像形状: {test_frame.shape}")
    print(f"测试图像类型: {test_frame.dtype}")
    
    # 单线程推理测试
    print("\n=== 单线程推理测试 ===")
    for i in range(3):
        try:
            start_time = time.time()
            result = shared_model.model(
                test_frame,
                conf=shared_model.conf_threshold,
                device=shared_model.device,
                imgsz=320,
                half=True,
                verbose=False
            )[0]
            end_time = time.time()
            print(f"第 {i+1} 次推理成功，耗时: {end_time - start_time:.2f}秒")
            print(f"推理结果类型: {type(result)}")
            print(f"推理结果框数量: {len(result.boxes)}")
        except Exception as e:
            print(f"第 {i+1} 次推理失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 多线程推理测试
    print("\n=== 多线程推理测试 ===")
    inference_count = 3
    thread_count = 2
    results = []
    lock = threading.Lock()
    
    def inference_thread(thread_id):
        print(f"线程 {thread_id} 开始执行...")
        thread_results = []
        for i in range(inference_count):
            try:
                start_time = time.time()
                result = shared_model.model(
                    test_frame,
                    conf=shared_model.conf_threshold,
                    device=shared_model.device,
                    imgsz=320,
                    half=True,
                    verbose=False
                )[0]
                end_time = time.time()
                thread_results.append((thread_id, i, "成功", end_time - start_time))
                print(f"线程 {thread_id}: 第 {i+1}/{inference_count} 次推理成功，耗时: {end_time - start_time:.2f}秒")
            except Exception as e:
                thread_results.append((thread_id, i, f"失败: {str(e)}", 0))
                print(f"线程 {thread_id}: 第 {i+1}/{inference_count} 次推理失败: {str(e)}")
                import traceback
                traceback.print_exc()
            time.sleep(0.1)
        with lock:
            results.extend(thread_results)
        print(f"线程 {thread_id} 完成执行")
    
    # 启动线程
    threads = []
    total_start_time = time.time()
    
    for i in range(thread_count):
        thread = threading.Thread(target=inference_thread, args=(i,))
        threads.append(thread)
        thread.start()
    
    # 等待线程完成
    for thread in threads:
        thread.join()
    
    total_end_time = time.time()
    
    # 统计结果
    print(f"\n=== 多线程推理结果统计 ===")
    print(f"总推理次数: {len(results)}")
    success_count = sum(1 for r in results if r[2] == "成功")
    failure_count = len(results) - success_count
    print(f"成功次数: {success_count}")
    print(f"失败次数: {failure_count}")
    print(f"总耗时: {total_end_time - total_start_time:.2f}秒")
    
    if success_count > 0:
        avg_time = sum(r[3] for r in results if r[2] == "成功") / success_count
        print(f"平均推理耗时: {avg_time:.2f}秒")
        print(f"平均每秒推理次数: {success_count / (total_end_time - total_start_time):.2f}")
    
    # 模型共享验证
    print("\n=== 模型共享验证 ===")
    model1 = DetectionModel(
        name="model1",
        model_path=model_path,
        target_classes=['bare'],
        conf_threshold=0.7
    )
    
    model2 = DetectionModel(
        name="model2",
        model_path=model_path,
        target_classes=['bare'],
        conf_threshold=0.7
    )
    
    is_shared = model1.model is model2.model
    print(f"模型1设备: {model1.device}")
    print(f"模型2设备: {model2.device}")
    print(f"两个模型实例是否共享同一个模型对象: {'✅ 是' if is_shared else '❌ 否'}")
    
    # 测试完成
    print("\n=== 测试完成 ===")
    if success_count == len(results) and is_shared:
        print("✅ 所有测试通过! 模型共享在多线程环境下正常工作。")
    else:
        print("❌ 部分测试失败，请检查问题。")
    
    # 释放资源
    print("\n释放资源...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"最终GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
    print("测试脚本执行完成!")
    
except Exception as e:
    print(f"\n❌ 测试过程中发生错误: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)