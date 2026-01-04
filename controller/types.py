import time
from typing import List
from ultralytics import YOLO

class DetectionResult:
    """单个检测结果"""
    def __init__(self, model_name: str, class_name: str, confidence: float, bbox: List[float]):
        self.model_name = model_name  # 'glove' 或 'head'
        self.class_name = class_name  # 'bare' 或 'touch'
        self.confidence = confidence
        self.bbox = [float(x) for x in bbox]  # [x1, y1, x2, y2]
    def bbox_equals(self, other_bbox: List[float], tol: float = 1e-3) -> bool:
        if other_bbox is None or len(other_bbox) != 4:
            return False
        return all(abs(float(a) - float(b)) <= tol for a, b in zip(self.bbox, other_bbox))


class DetectionHistory:
    """管理单一异常类的历史检测状态"""
    def __init__(self, history_refresh_time: float = 1800.0):  # 默认30分钟刷新一次
        self.history_boxes = []  # 历史检测框列表 [bbox]
        self.last_refresh_time = time.time()
        self.history_refresh_time = history_refresh_time  # 历史记录刷新时间（秒）
    
    def add_box(self, bbox: List[float]):
        """添加新的检测框到历史记录"""
        self.history_boxes.append(bbox)
        self._check_and_refresh()
    
    def _check_and_refresh(self):
        """检查并刷新历史记录"""
        current_time = time.time()
        if current_time - self.last_refresh_time > self.history_refresh_time:
            # print(f"\n=== 历史记录刷新 ===")
            # print(f"刷新历史记录，删除了 {len(self.history_boxes)} 个框")
            # print(f"====================")
            self.history_boxes = []
            self.last_refresh_time = current_time
            # print(f"历史记录已刷新，当前时间: {time.ctime()}")
    
    @staticmethod
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        """计算两个bbox的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        # 计算交集面积
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0
        # 计算两个框的面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        # 计算IoU
        iou = intersection / (area1 + area2 - intersection)
        return iou
    
    def is_new_detection(self, bbox: List[float], scale_factor: float = 2.0) -> bool:
        """检查新检测框是否与历史框匹配
        Args:
            bbox: 新检测框 [x1, y1, x2, y2]
            scale_factor: 历史框的放大倍数，默认2倍
        Returns:
            bool: 如果新框不在任何历史框的scale_factor倍范围内，则返回True
        """
        self._check_and_refresh()
        
        if not self.history_boxes:
            return True
        
        for history_box in self.history_boxes:
            # 计算历史框的中心点
            h_center_x = (history_box[0] + history_box[2]) / 2
            h_center_y = (history_box[1] + history_box[3]) / 2
            h_width = history_box[2] - history_box[0]
            h_height = history_box[3] - history_box[1]
            
            # 计算放大后的历史框
            scaled_h_box = [
                int(h_center_x - (h_width * scale_factor) / 2),
                int(h_center_y - (h_height * scale_factor) / 2),
                int(h_center_x + (h_width * scale_factor) / 2),
                int(h_center_y + (h_height * scale_factor) / 2)
            ]
            
            # 计算IoU
            iou = DetectionHistory._calculate_iou(bbox, scaled_h_box)
            
            # 如果IoU大于0，说明新框在放大后的历史框范围内
            if iou > 0:
                return False
        
        return True


from typing import List, Dict, Any
from ultralytics import YOLO
import torch
import threading

# 模型管理器单例类，用于共享模型实例
class ModelManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance.models = {}
                    cls._instance.model_lock = threading.Lock()
                    # 添加全局推理锁，确保同一时间只有一个线程在使用模型进行推理
                    cls._instance.inference_global_lock = threading.Lock()
        return cls._instance
    
    def get_model(self, model_path: str, device: str = 'cuda'):
        """获取或创建模型实例"""
        with self.model_lock:
            if model_path not in self.models:
                # 加载新模型前先清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # 打印内存使用情况
                    print(f"[DEBUG] 加载模型前 GPU内存使用: {torch.cuda.memory_allocated()/1024/1024:.2f} MB, 缓存: {torch.cuda.memory_reserved()/1024/1024:.2f} MB")
                
                # 尝试使用更小的模型配置加载
                try:
                    # 降低模型加载时的内存需求
                    if device == 'cuda':
                        # 先在CPU上加载模型，然后再转移到GPU
                        model = YOLO(model_path)
                        print(f"[DEBUG] 模型在CPU上加载成功，准备转移到GPU")
                        model.to(device)
                    else:
                        model = YOLO(model_path)
                        model.to(device)
                    
                    self.models[model_path] = model
                    
                    # 打印内存使用情况
                    if torch.cuda.is_available():
                        print(f"[DEBUG] 加载模型后 GPU内存使用: {torch.cuda.memory_allocated()/1024/1024:.2f} MB, 缓存: {torch.cuda.memory_reserved()/1024/1024:.2f} MB")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # 如果内存不足，尝试更激进的内存优化
                        print(f"[DEBUG] 内存不足，尝试更激进的内存优化")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        # 强制使用CPU
                        print(f"[DEBUG] 强制使用CPU加载模型")
                        model = YOLO(model_path)
                        model.to('cpu')
                        self.models[model_path] = model
                        device = 'cpu'  # 更新设备为CPU
                    else:
                        raise
            return self.models[model_path]
    
    def release_all_models(self):
        """释放所有模型占用的GPU内存"""
        with self.model_lock:
            for model_path, model in self.models.items():
                if hasattr(model, 'cpu'):
                    try:
                        model.cpu()
                    except Exception:
                        pass
            self.models.clear()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

class DetectionModel:
    """检测模型，封装 YOLO 模型"""
    def __init__(self,
                 name: str,
                 model_path: str,
                 target_classes: List[str],
                 conf_threshold: float = 0.5,
                 frame_threshold: int = 2,
                 trigger_mode: str = "area",
                 enabled: bool = True):  # trigger_mode: "area" or "any"
        """
        name: 模型 key, 如 'glove' / 'head'
        trigger_mode:
            - 'area' : 只有 bbox 完全进入 area 区域才计为危险（适合 glove）
            - 'any'  : 只要检测到就计为危险（适合 touch）
        """
        self.name = name
        self.model_path = model_path
        self.target_classes = target_classes
        self.conf_threshold = conf_threshold
        self.frame_threshold = frame_threshold
        self.trigger_mode = trigger_mode
        self.enabled = enabled  # 可由外部配置开关
        
        # 检查CUDA可用性并设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 使用模型管理器获取共享模型实例
        self.model_manager = ModelManager()
        self.model = self.model_manager.get_model(model_path, self.device)
        
        # 确保模型处于评估模式
        self.model.eval()
        
        # 设置BatchNorm层为推理模式，避免多线程问题
        for module in self.model.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                module.track_running_stats = False
        
        # 更新设备为模型实际使用的设备
        # 检查模型的设备
        if hasattr(self.model, 'device'):
            self.device = self.model.device
        else:
            # 对于YOLO模型，检查模型的预测头设备
            try:
                first_param = next(self.model.parameters())
                self.device = first_param.device
            except StopIteration:
                pass  # 如果没有参数，保持原设备设置
        
        print(f"[DEBUG] {self.name} 模型设备: {self.device}")
    def release(self):
        """释放模型占用的GPU内存（实际由ModelManager管理）"""
        # 这里不需要释放模型，因为模型是共享的
        # 只有当所有DetectionModel实例都不再使用时，才会由ModelManager统一释放
        pass