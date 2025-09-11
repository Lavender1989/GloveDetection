"""
存储视频/监控与视角的对应关系
"""

import re
import os

# 定义视频名称与视角的映射关系，使用列表存储不同视角对应的关键字
# 这样后续只需要在列表中添加新的关键字即可
VIEW_1_KEYWORDS = ['20250829_1', '20250820_1']
VIEW_2_KEYWORDS = ['20250829_2', '20250820_2']


def get_view_for_video(video_path):
    """
    根据视频路径确定对应的视角
    Args:
        video_path: 视频文件路径或RTSP地址
    Returns:
        int: 视角索引 (0表示视角1, 1表示视角2)
    """
    video_name = os.path.basename(video_path)
    
    # RTSP地址处理
    if video_path.lower().startswith("rtsp://"):
        # 提取IP地址的最后一段数字
        ip_match = re.search(r'\b(?:\d{1,3}\.){3}(\d{1,3})\b', video_path)
        if ip_match:
            last_octet = ip_match.group(1)
            # 102对应视角1, 106对应视角2
            if last_octet == "102":
                return 0
            elif last_octet == "106":
                return 1
        # 默认视角1
        return 0
    
    # 本地文件处理 - 使用列表方式判断
    # 检查视频名称是否包含视角1的关键字
    for keyword in VIEW_1_KEYWORDS:
        if keyword in video_name:
            return 0
    # 检查视频名称是否包含视角2的关键字
    for keyword in VIEW_2_KEYWORDS:
        if keyword in video_name:
            return 1
    
    # 默认视角1
    return 0

def get_view_name(view_index):
    """获取视角名称"""
    view_names = ["视角1", "视角2"]
    if 0 <= view_index < len(view_names):
        return view_names[view_index]
    return "未知视角"