import torch
from ultralytics import YOLO

model = YOLO("/home/test/Desktop/GloveDetection/model/glove/best1219.pt")  # 或你的模型路径
model.to("cuda")
print("model loaded!")
