from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel

from .main_ui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self, controller=None):  # 接收控制器实例
        super().__init__()
        # 初始化UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # 保存控制器引用（关键：之前缺少控制器初始化）
        self.controller = controller

        # 暴露界面元素
        self.choose_scene = self.ui.chooseScene
        self.add_scene_btn = self.ui.addScene
        self.delete_scene_btn = self.ui.deleteScene
        self.add_video_btn = self.ui.addVideo
        self.delete_video_btn = self.ui.deleteVideo
        self.edit_video_btn = self.ui.editVideo
        self.video_list = self.ui.videoList
        self.video_display = self.ui.videoDisplay  # 这是QTabWidget控件
        self.log_box = self.ui.logBox
        self.start_detection_btn = self.ui.startDetection
        self.close_detection_btn = self.ui.closeDetection

        # 视频标签页管理字典（唯一字典，存储所有标签页信息）
        self.video_tabs = {}  # 格式: {video_id: {'widget': QWidget, 'label': QLabel, 'index': int}}

        # 初始化视频显示标签页
        self.init_video_tabs()

        # 连接控制器的信号到UI更新函数
        # 仅在控制器存在时连接信号
        # if self.controller:
        #     self.controller.video_frame_updated.connect(self.update_video_frame)
        #     self.controller.video_added.connect(self.add_video_tab)
        #     self.controller.video_removed.connect(self.remove_video_tab)

    def init_video_tabs(self):
        """初始化视频显示标签页控件（配置QTabWidget）"""
        self.video_display.setMovable(True)  # 允许标签页拖动
        self.video_display.setTabsClosable(True)  # 允许关闭标签页
        self.video_display.tabCloseRequested.connect(self.on_video_tab_closed)

    def add_video_tab(self, video_id, video_name):
        """添加新的视频标签页"""
        if video_id in self.video_tabs:
            return

        # 创建标签页内容
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)

        # 创建视频显示标签
        video_label = QLabel()
        video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_label.setStyleSheet("background-color: black;")
        layout.addWidget(video_label)

        # 添加到QTabWidget控件（注意：这里必须用self.video_display）
        index = self.video_display.addTab(tab_widget, video_name)
        self.video_display.setCurrentIndex(index)

        # 存储标签页信息到字典
        self.video_tabs[video_id] = {
            'widget': tab_widget,
            'label': video_label,
            'index': index
        }

    def remove_video_tab(self, video_id):
        """移除视频标签页"""
        if video_id not in self.video_tabs:
            return

        # 从QTabWidget中移除标签页
        index = self.video_tabs[video_id]['index']
        self.video_display.removeTab(index)
        del self.video_tabs[video_id]

        # 更新剩余标签页的索引
        self._update_video_tab_indices()

    def _update_video_tab_indices(self):
        """更高效的索引更新方式"""
        for video_id in list(self.video_tabs.keys()):  # 遍历副本避免修改时出错
            widget = self.video_tabs[video_id]['widget']
            index = self.video_display.indexOf(widget)  # 直接通过widget获取索引
            if index != -1:
                self.video_tabs[video_id]['index'] = index

    def _get_video_id_from_tab_index(self, index):
        """从标签页索引获取对应的video_id"""
        widget = self.video_display.widget(index)
        for video_id, info in self.video_tabs.items():
            if info['widget'] == widget:
                return video_id
        return None

    def on_video_tab_closed(self, index):
        """处理标签页关闭事件"""
        video_id = self._get_video_id_from_tab_index(index)
        if video_id and self.controller:  # 添加controller存在性检查
            self.controller.stop_video_detection(video_id)
            self.remove_video_tab(video_id)
        elif video_id:
            # 如果没有控制器，直接移除标签页
            self.remove_video_tab(video_id)

    def update_video_frame(self, video_id, qimage):
        """更新指定视频标签页的画面"""
        if video_id not in self.video_tabs:
            return

        label = self.video_tabs[video_id]['label']
        pixmap = QPixmap.fromImage(qimage)

        # 保持比例缩放
        pixmap = pixmap.scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(pixmap)