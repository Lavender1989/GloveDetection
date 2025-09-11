from ultralytics import YOLO
import cv2
import os
import xml.etree.ElementTree as ET
from collections import deque
import time
import tkinter as tk
from tkinter import messagebox
import threading
import argparse
class VideoDetector:
    def __init__(self, video_path, model_path, output_dir=None, alert_threshold=5, alert_duration=3):
        self.ALERT_FRAME_THRESHOLD = alert_threshold
        self.ALERT_DISPLAY_SECONDS = alert_duration
        self.consecutive_danger_frames = 0 
        self.alert_active = False
        self.alert_start_time = 0
        self.area_boxes = []  
        self.current_view = 0 
        self.view_names = ["视角1", "视角2"]
        self.xml_paths = [
            r'd:\detect_video\area\20250829_1_frame00000.xml', 
            r'd:\detect_video\area\20250829_2_frame00000.xml' 
        ]
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.current_view = self.get_view_by_video_name(video_path)
        self.area_boxes = self.load_area_from_xml(self.xml_paths[self.current_view])
        print(f"已加载 {self.view_names[self.current_view]} 对应的区域配置")
        self.model = YOLO(model_path)
        print(f"已加载模型: {model_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise Exception(f"无法打开视频源: {self.video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        filename = os.path.basename(self.video_path)
        self.output_path = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (960, 540))
        if not self.out.isOpened():
            raise Exception(f"无法打开视频写入器，请检查编码器和输出路径: {self.output_path}")
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection", 960, 540)
    def load_area_from_xml(self, xml_path):
        """从XML文件加载检测区域"""
        area_boxes = []
        try:
            if not os.path.exists(xml_path):
                print(f"XML文件不存在: {xml_path}")
                return area_boxes
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name == 'area': 
                    bndbox = obj.find('bndbox')
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    area_boxes.append([xmin, ymin, xmax, ymax])
            print(f"成功从 {xml_path} 加载了 {len(area_boxes)} 个检测区域")
        except Exception as e:
            print(f"加载XML文件 {xml_path} 时出错: {e}")
        return area_boxes
    def show_warning_popup(self):
        """显示警告弹窗"""
        def _popup():
            root = tk.Tk()
            root.withdraw() 
            messagebox.showwarning("警告", "检测到未佩戴手套操作！")
            root.destroy()

        threading.Thread(target=_popup).start()
    def draw_detections(self, frame, bare_boxes, danger_boxes):
        for box in self.area_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "area", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for box in bare_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "bare", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        for box in danger_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "DANGER", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    def box_intersection(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        return (xi1 < xi2) and (yi1 < yi2)
    def box_fully_contains(self, box1, box2):
        """检查box2是否完全包含在box1内"""
        x1, y1, x2, y2 = box1  # 区域框
        x3, y3, x4, y4 = box2  # bare框
        result = (x3 >= x1) and (y3 >= y1) and (x4 <= x2) and (y4 <= y2)
        print(f"检查区域框 ({x1}, {y1}, {x2}, {y2}) 是否完全包含bare框 ({x3}, {y3}, {x4}, {y4}): {result}")
        return result
    def get_view_by_video_name(self, video_path):
        """根据视频文件名确定使用哪个视角"""
        filename = os.path.basename(video_path)
        if '20250829_1' in filename:
            return 0  # 视角1
        elif '20250829_2' in filename:
            return 1  # 视角2
        else:
            print(f"警告：无法识别视频文件 {filename} 对应的视角，默认使用视角1")
            return 0  # 默认使用视角1
    def process_video(self):
        """处理视频并进行检测"""
        frame_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if frame is None:
                print(f"读取到空帧，帧索引: {frame_count}")
            if not ret:
                print("视频读取结束或失败")
                break
            class_names = self.model.names
            bare_class_id = None
            for class_id, name in class_names.items():
                if name == 'bare':
                    bare_class_id = class_id
                    break
            if bare_class_id is not None:
                results = self.model(frame, conf=0.8, classes=[bare_class_id])[0]
            else:
                results = self.model(frame, conf=0.8)[0]
            annotated_frame = frame.copy()
            danger_detected = False
            bare_boxes = []
            danger_boxes = []
            detected_area_indices = []  # 记录检测到危险的区域索引
            for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls):
                cls = int(cls)
                if results.names[cls] == 'bare':
                    bare_boxes.append(box)
            for bare_box_idx, bare_box in enumerate(bare_boxes):
                for area_idx, area_box in enumerate(self.area_boxes):
                    if self.box_fully_contains(area_box, bare_box):
                        danger_detected = True
                        danger_boxes.append(bare_box)
                        detected_area_indices.append(area_idx)
                        break
            if danger_detected:
                self.consecutive_danger_frames += 1
                unique_areas = set(detected_area_indices)
                area_info = f"区域{', '.join(map(str, unique_areas))}"
                print(f"帧 {frame_count}: 在{area_info} 检测到未佩戴手套 (连续危险帧: {self.consecutive_danger_frames})")
            else:
                if self.consecutive_danger_frames > 0:
                    print(f"帧 {frame_count}: 未检测到危险，重置连续危险帧计数 (之前计数: {self.consecutive_danger_frames})")
                self.consecutive_danger_frames = 0
            current_time = time.time()
            if len(danger_boxes) > 0:
                if not self.alert_active:
                    self.consecutive_danger_frames += 1
                    print(f"[DEBUG] 连续危险帧数: {self.consecutive_danger_frames}")
                if self.consecutive_danger_frames >= self.ALERT_FRAME_THRESHOLD and not self.alert_active:
                    self.alert_active = True
                    self.alert_start_time = current_time
                    print(f"[ALARM] 检测到 bare 类别完全进入危险区域！连续危险帧数: {self.consecutive_danger_frames}")
            else:
                if not self.alert_active:
                    self.consecutive_danger_frames = 0
                    print("[DEBUG] 无危险框，重置连续危险帧计数")
            if self.alert_active:
                alert_duration = current_time - self.alert_start_time
                cv2.putText(annotated_frame, "ALARM! BARE HANDS DETECTED!", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Duration: {int(alert_duration)}s", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if alert_duration >= self.ALERT_DISPLAY_SECONDS:
                    print("[ALARM RESET] 报警时间已达阈值，重置报警状态和计数")
                    self.alert_active = False
                    self.alert_start_time = None
                    self.consecutive_danger_frames = 0  # 清空连续危险帧计数
            if self.alert_active:
                text = "warning"
                color = (0, 0, 255)
            else:
                text = "normal"
                color = (0, 255, 0)
            cv2.putText(annotated_frame, f"视角: {self.view_names[self.current_view]}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(annotated_frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            self.draw_detections(annotated_frame, bare_boxes, danger_boxes)
            annotated_frame = cv2.resize(annotated_frame, (960, 540))
            cv2.imshow("Detection", annotated_frame)
            if annotated_frame.shape[1] == 960 and annotated_frame.shape[0] == 540:
                self.out.write(annotated_frame)
                frame_count += 1
                print(f"成功写入帧 {frame_count}")
            else:
                print(f"跳过尺寸不匹配的帧: {annotated_frame.shape[1]}x{annotated_frame.shape[0]}")
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        self.cap.release()
        print(f"视频处理完成: 共尝试写入 {frame_count} 帧\n输出文件路径: {self.output_path}\n文件大小: {os.path.getsize(self.output_path)} 字节")
        self.out.release()
        cv2.destroyAllWindows()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='视频检测脚本')
    parser.add_argument('--video', type=str, default=r"d:\detect_video\original_Video\no_wearing_gloves\new\20250829_1.mp4", help='视频文件路径')
    parser.add_argument('--model', type=str, default=r"d:\detect_video\ultralytics-main\experiment\gloves_detection0909\weights\best.pt", help='模型权重文件路径')
    parser.add_argument('--output_dir', type=str, default=r'd:\detect_video\detected_video', help='输出视频目录')
    parser.add_argument('--alert_threshold', type=int, default=10, help='连续异常帧数阈值')
    parser.add_argument('--alert_duration', type=int, default=3, help='报警信息显示时间(秒)')
    args = parser.parse_args()

    try:
        # 创建检测器实例
        detector = VideoDetector(
            video_path=args.video,
            model_path=args.model,
            output_dir=args.output_dir,
            alert_threshold=args.alert_threshold,
            alert_duration=args.alert_duration
        )
        # 处理视频
        detector.process_video()
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()