"""
存储视频/监控与视角的对应关系
"""

import os
import xml.etree.ElementTree as ET

# 定义视频名称与视角的映射关系，使用列表存储不同视角对应的关键字
# 这样后续只需要在列表中添加新的关键字即可
VIEW_1_KEYWORDS = ['20250829_1', '20250820_1', '20250729_2', '20250729_1', '1204(1)',
                    '0911_1', '1112_1',
                   'rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/201']
VIEW_2_KEYWORDS = ['20250829_2', '20250820_2', '1204', '1112_3', '1112_2', '1120_3',
                    '1120_4', '1120_5(2)', '1120_6(2)', '1120_7', '1120_8',
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
        # 首先检查是否完全匹配关键字列表中的URL
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


def view2xml(view_index, width=None, height=None):
    xml_paths = [
        os.path.join(os.path.dirname(__file__), "..", "area", "0911_1_frame00000.xml"),  # VIEW_1
        os.path.join(os.path.dirname(__file__), "..", "area", "0911_2_frame00000.xml"), # VIEW_2
        os.path.join(os.path.dirname(__file__), "..", "area", "301.xml"), # VIEW_3
        os.path.join(os.path.dirname(__file__), "..", "area", "401.xml"), # VIEW_4
        os.path.join(os.path.dirname(__file__), "..", "area", "501.xml"), # VIEW_5
        os.path.join(os.path.dirname(__file__), "..", "area", "601.xml"), # VIEW_6
        os.path.join(os.path.dirname(__file__), "..", "area", "701.xml"), # VIEW_7
        os.path.join(os.path.dirname(__file__), "..", "area", "901.xml"), # VIEW_8
        os.path.join(os.path.dirname(__file__), "..", "area", "1201.xml"), # VIEW_9
        os.path.join(os.path.dirname(__file__), "..", "area", "1301.xml"), # VIEW_10
        ]
    if 0 <= view_index < len(xml_paths):
        xml_path = xml_paths[view_index]
        # 先记录加载信息
        load_message = f"加载区域: {xml_path}"
        area_boxes = load_area_from_xml(xml_path, width, height)
        log_message = f"{load_message}，加载到 {len(area_boxes)} 个区域"
    else:
        area_boxes = []
        log_message= "没有对应视角，未加载区域"
    return area_boxes, log_message

def load_area_from_xml(xml_path, width=None, height=None):
    area_boxes = []
    if not os.path.exists(xml_path):
        # 返回空列表
        return area_boxes
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_node = root.find("size")
        xml_w = int(size_node.find("width").text) if size_node is not None and size_node.find("width") is not None else None
        xml_h = int(size_node.find("height").text) if size_node is not None and size_node.find("height") is not None else None
        raw = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name == 'area':
                bnd = obj.find('bndbox')
                xmin = int(float(bnd.find('xmin').text))
                ymin = int(float(bnd.find('ymin').text))
                xmax = int(float(bnd.find('xmax').text))
                ymax = int(float(bnd.find('ymax').text))
                raw.append([xmin, ymin, xmax, ymax])
        # 缩放
        if width is None or height is None:
            return [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in raw]
        tw, th = int(width), int(height)
        if xml_w and xml_h:
            sx = tw / xml_w
            sy = th / xml_h
            for x1, y1, x2, y2 in raw:
                nx1 = max(0, min(tw - 1, int(round(x1 * sx))))
                ny1 = max(0, min(th - 1, int(round(y1 * sy))))
                nx2 = max(0, min(tw - 1, int(round(x2 * sx))))
                ny2 = max(0, min(th - 1, int(round(y2 * sy))))
                area_boxes.append([nx1, ny1, nx2, ny2])
        return area_boxes
    except Exception as e:
        # 解析失败时返回空列表
        return []

if __name__ == "__main__":
    video_path = 'rtsp://admin:abc12345@10.66.3.243:554/Streaming/Channels/201'
    view_index = get_view_for_video(video_path)
    print(view_index)