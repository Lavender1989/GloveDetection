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


class DetectionModel:
    """包含模型、阈值与触发策略"""
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
        self.model = YOLO(model_path)
        # 检查CUDA可用性并设置设备
        self.device = 'cuda'
        self.model.to(self.device)
        # print(f"{self.name} 模型已加载到 {self.device} 设备")
        self.target_classes = target_classes
        self.conf_threshold = conf_threshold
        self.frame_threshold = frame_threshold
        self.trigger_mode = trigger_mode
        self.enabled = enabled  # 可由外部配置开关
    
    def release(self):
        """释放模型占用的GPU内存"""
        if hasattr(self, 'model'):
            # 尝试释放模型
            import torch
            try:
                # 清除模型占用的GPU内存
                self.model.cpu()  # 将模型移回CPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                pass
            finally:
                # 置空模型引用
                self.model = None