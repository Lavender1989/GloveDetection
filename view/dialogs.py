import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QRadioButton, QGroupBox,
                             QFileDialog, QComboBox, QMessageBox, QListWidget,
                             QListWidgetItem, QAbstractItemView)

class VideoSourceDialog(QDialog):
    """添加/编辑视频源对话框"""

    def __init__(self, parent=None, video_info=None, scene_id=None):
        super().__init__(parent)
        self.setWindowTitle("添加视频源" if not video_info else "编辑视频源")
        self.resize(400, 300)

        # 初始化属性
        self.video_info = video_info
        self.scene_id = scene_id
        self.selected_type = 1  # 1:本地文件 2:RTSP 3:摄像头

        self.init_ui()
        self._init_edit_mode()  # 单独拆分编辑模式初始化

    def init_ui(self):
        """初始化UI布局"""
        main_layout = QVBoxLayout()

        # 1. 名称输入
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("视频名称:"))
        self.name_input = QLineEdit()
        name_layout.addWidget(self.name_input)
        main_layout.addLayout(name_layout)

        # 2. 视频类型选择
        type_group = QGroupBox("视频源类型")
        type_layout = QVBoxLayout()

        self.local_radio = QRadioButton("本地视频文件")
        self.local_radio.setChecked(True)
        self.local_radio.toggled.connect(lambda: self.on_type_changed(1))

        self.rtsp_radio = QRadioButton("RTSP地址")
        self.rtsp_radio.toggled.connect(lambda: self.on_type_changed(2))

        self.camera_radio = QRadioButton("本机摄像头")
        self.camera_radio.toggled.connect(lambda: self.on_type_changed(3))

        type_layout.addWidget(self.local_radio)
        type_layout.addWidget(self.rtsp_radio)
        type_layout.addWidget(self.camera_radio)
        type_group.setLayout(type_layout)
        main_layout.addWidget(type_group)

        # 3. 路径输入
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("路径/地址:"))
        self.path_input = QLineEdit()
        self.browse_btn = QPushButton("浏览")
        self.browse_btn.clicked.connect(self.browse_path)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_btn)
        main_layout.addLayout(path_layout)

        # 4. 邮箱选择（新增）支持多选
        email_group = QGroupBox("报警邮箱 (可多选)")
        email_group_layout = QVBoxLayout()
        
        # 创建邮箱列表控件
        from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
        self.email_list = QListWidget()
        self.email_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        
        # 所有可用邮箱列表
        self.available_emails = [
            {"name": "管理员1", "email": "903466339@qq.com"},
            {"name": "管理员2", "email": "Honglingxiang@kaifa.cn"},
            {"name": "管理员3", "email": "XinHuZhang@kaifa.cn"},
            {"name": "管理员4", "email": "ShaoHuawang1@kaifa.cn"},
            {"name": "管理员5", "email": "xiaoyuzhong@kaifa.cn"},
            {"name": "管理员6", "email": "wqr20011989@163.com"}
        ]
        
        # 添加邮箱到列表
        for email_info in self.available_emails:
            item = QListWidgetItem(f"{email_info['name']} ({email_info['email']})")
            item.setData(Qt.ItemDataRole.UserRole, email_info['email'])
            self.email_list.addItem(item)
        
        email_group_layout.addWidget(self.email_list)
        email_group.setLayout(email_group_layout)
        main_layout.addWidget(email_group)

        # 5. 底部按钮
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("确定")
        self.cancel_btn = QPushButton("取消")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def _init_edit_mode(self):
        """初始化编辑模式（单独拆分，减少__init__复杂度）"""
        if not self.video_info:
            return

        # 填充已有信息
        self.name_input.setText(self.video_info.name)
        self.path_input.setText(self.video_info.path)
        self.selected_type = self.video_info.type

        # 切换单选按钮状态
        if self.selected_type == 1:
            self.local_radio.setChecked(True)
        elif self.selected_type == 2:
            self.rtsp_radio.setChecked(True)
        else:
            self.camera_radio.setChecked(True)
            
        # 设置邮箱选择（支持多选）
        if hasattr(self.video_info, 'alert_email') and self.video_info.alert_email:
            # 将存储的邮箱字符串分割成列表
            if isinstance(self.video_info.alert_email, str):
                selected_emails = [email.strip() for email in self.video_info.alert_email.split(',')]
            else:
                selected_emails = [self.video_info.alert_email]
                
            # 遍历列表并选中对应的邮箱
            for i in range(self.email_list.count()):
                item = self.email_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) in selected_emails:
                    item.setSelected(True)

    """视频类型切换处理"""
    def on_type_changed(self, type_id):
        self.selected_type = type_id
        # 更新浏览按钮文本
        btn_text = {1: "浏览", 2: "输入", 3: "选择设备"}
        self.browse_btn.setText(btn_text[type_id])

    """根据类型处理路径选择/测试"""
    def browse_path(self):
        if self.selected_type == 1:
            self._select_local_file()
        elif self.selected_type == 2:
            pass
        elif self.selected_type == 3:
            self._select_camera()

    """获取选择的邮箱列表（多个邮箱用逗号分隔）"""
    def get_selected_email(self):
        selected_items = self.email_list.selectedItems()
        if not selected_items:
            # 如果没有选择，返回默认的第一个邮箱
            return self.available_emails[0]['email'] if self.available_emails else ""
            
        # 收集所有选中的邮箱
        selected_emails = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        # 用逗号分隔存储多个邮箱
        return ",".join(selected_emails)

    """选择本地视频文件"""
    def _select_local_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.path_input.setText(file_path)

    """选择摄像头设备"""
    def _select_camera(self):
        available_cameras = []
        # 检测前10个设备ID
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()

        if not available_cameras:
            QMessageBox.warning(self, "警告", "未检测到可用摄像头")
            return

        # 摄像头选择对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择摄像头")
        layout = QVBoxLayout()

        camera_combo = QComboBox()
        for cam_id in available_cameras:
            camera_combo.addItem(f"摄像头 {cam_id}", cam_id)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        ok_btn.clicked.connect(lambda: self._confirm_camera(camera_combo, dialog))
        cancel_btn.clicked.connect(dialog.reject)

        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addWidget(QLabel("请选择摄像头:"))
        layout.addWidget(camera_combo)
        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        dialog.exec()

    """确认选择的摄像头"""
    def _confirm_camera(self, combo, dialog):
        selected_id = combo.currentData()
        self.path_input.setText(str(selected_id))
        dialog.accept()

    """确认按钮处理"""
    def accept(self):
        """确认按钮处理"""
        # 移除RTSP测试验证逻辑
        if not self.name_input.text().strip() or not self.path_input.text().strip():
            QMessageBox.warning(self, "警告", "名称和路径不能为空")
            return
        super().accept()

    """返回视频源信息,加入数据库"""
    def get_video_info(self):
        return {
            "name": self.name_input.text(),
            "path": self.path_input.text(),
            "type": self.selected_type,
            "scene_id": self.scene_id,
            "is_true": False,
            "is_valid": True,
            "alert_email": self.get_selected_email()  # 新增邮箱信息
        }


class SceneDialog(QDialog):
    """添加/编辑场景对话框"""

    def __init__(self, parent=None, scene_name=""):
        super().__init__(parent)
        self.setWindowTitle("添加场景" if not scene_name else "编辑场景")
        self.resize(300, 120)
        self.scene_name = scene_name
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("场景名称:"))
        self.name_input = QLineEdit(self.scene_name)
        layout.addWidget(self.name_input)

        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("确定")
        self.cancel_btn = QPushButton("取消")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def get_scene_name(self):
        return self.name_input.text().strip()