"""
测试video_view_mapping.py的功能
"""
import os
import sys
from dataclasses import dataclass

# 添加项目根目录到Python路径，确保能正确导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入需要测试的模块
from controller.video_view_mapping import get_view_for_video, get_view_name

# 模拟VideoSource类，与db.py中的定义保持一致
@dataclass
class MockVideoSource:
    id: int
    name: str
    path: str
    is_true: bool = True
    is_valid: bool = True
    scene_id: int = 1
    type: int = 1  # 1:本地文件, 2:RTSP, 3:摄像头
    alert_email: str = None

# 模拟日志函数
def mock_log(message):
    timestamp = "12:34:56"  # 模拟时间戳
    print(f"[{timestamp}] {message}")

# 模拟DetectionThread中获取视角的流程
def test_view_detection(video_source):
    mock_log(f"开始处理视频: {video_source.name}")
    
    # 日志输出视频类型
    video_type = "RTSP地址" if video_source.path.lower().startswith("rtsp://") else "本地文件"
    mock_log(f"视频类型: {video_type}")
    
    # 提前确定视角并输出日志
    view_index = get_view_for_video(video_source.path)
    view_name = get_view_name(view_index)
    mock_log(f"{video_source.name}: 成功加载{view_name}")
    
    return view_index

# 测试用例
def run_tests():
    print("========== 视频视角映射测试 ==========")
    
    # 测试RTSP视频源
    test_cases = [
        # RTSP测试用例
        # MockVideoSource(1, "RTSP摄像头1", "rtsp://admin:password@192.168.1.102:554/stream1", type=2),
        # MockVideoSource(2, "RTSP摄像头2", "rtsp://admin:password@192.168.1.106:554/stream1", type=2),
        # MockVideoSource(3, "未知RTSP摄像头", "rtsp://admin:password@192.168.1.103:554/stream1", type=2),
        
        # 本地文件测试用例
        MockVideoSource(4, "本地视频1", "D:\\videos\\20250829_1.mp4", type=1),
        MockVideoSource(5, "本地视频2", "D:\\videos\\20250829_2.mp4", type=1),
        MockVideoSource(6, "video2", "D:\\videos\\20250820_1.mp4", type=1),
        MockVideoSource(7, "video3", "D:\\videos\\20250820_2.mp4", type=1)
    ]
    
    for i, video in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {video.name}")
        print(f"路径: {video.path}")
        view_index = test_view_detection(video)
        print(f"检测结果: 视角索引={view_index}, 视角名称={get_view_name(view_index)}")

if __name__ == "__main__":
    run_tests()