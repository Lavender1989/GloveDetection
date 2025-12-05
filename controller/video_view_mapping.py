"""
存储视频/监控与视角的对应关系
"""

import re
import os

from cv2.gapi import video

# 定义视频名称与视角的映射关系，使用列表存储不同视角对应的关键字
# 这样后续只需要在列表中添加新的关键字即可
VIEW_1_KEYWORDS = ['20250829_1', '20250820_1', 
                   'rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/201']
VIEW_2_KEYWORDS = ['20250829_2', '20250820_2',
                   'rtsp://admin:abc12345@192.168.1.102/cam/realmonitor?channel=1&subtype=0&proto=Private3']
VIEW_3_KEYWORDS = ['rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/301']
VIEW_4_KEYWORDS = ['rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/401']
VIEW_5_KEYWORDS = ['rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/501']
VIEW_6_KEYWORDS = ['rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/601']
VIEW_7_KEYWORDS = ['rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/701']
VIEW_8_KEYWORDS = ['rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/901']
VIEW_9_KEYWORDS = ['rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/1201']
VIEW_10_KEYWORDS = ['rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/1301']

def get_view_for_video(video_path):
    """
    根据视频路径确定对应的视角
    Args:
        video_path: 视频文件路径或RTSP地址
    Returns:
        int: 视角索引 (0表示视角1, 1表示视角2,... -1表示未找到对应视角)
    """
    video_name = os.path.basename(video_path)
    
    # RTSP地址处理
    if video_path.lower().startswith("rtsp://"):
        # 提取IP地址的最后一段数字
        # ip_match = re.search(r'\b(?:\d{1,3}\.){3}(\d{1,3})\b', video_path)
        # if ip_match:
        #     last_octet = ip_match.group(1)
        #     # 104对应视角1, 102对应视角2
        #     # 改了一个对应的关系：rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/201 对应视角1
        #     if last_octet == "201":
        #         return 0
        #     elif last_octet == "102":
        #         return 1
        if video_path in VIEW_1_KEYWORDS:
            return 0
        elif video_path in VIEW_2_KEYWORDS:
            return 1
        elif video_path in VIEW_3_KEYWORDS:
            return 2
        elif video_path in VIEW_4_KEYWORDS:
            return 3
        elif video_path in VIEW_5_KEYWORDS:
            return 4
        elif video_path in VIEW_6_KEYWORDS:
            return 5
        elif video_path in VIEW_7_KEYWORDS:
            return 6
        elif video_path in VIEW_8_KEYWORDS:
            return 7
        elif video_path in VIEW_9_KEYWORDS:
            return 8
        elif video_path in VIEW_10_KEYWORDS:
            return 9
        # 默认全局视角
        return -1
    
    # 本地文件处理 - 使用列表方式判断
    # 检查视频名称是否包含视角1的关键字
    for keyword in VIEW_1_KEYWORDS:
        if keyword in video_name:
            return 0
    # 检查视频名称是否包含视角2的关键字
    for keyword in VIEW_2_KEYWORDS:
        if keyword in video_name:
            return 1
    
    # 默认没有视角
    return -1   

def get_view_name(view_index):
    """获取视角名称"""
    view_names = ["视角1", "视角2", "视角3", "视角4", "视角5", "视角6", "视角7", "视角8", "视角9", "视角10"]
    if 0 <= view_index < len(view_names):
        return view_names[view_index]
    return "全局视角"

if __name__ == "__main__":
    video_path = 'rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/201'
    view_index = get_view_for_video(video_path)
    print(view_index)