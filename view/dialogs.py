import cv2
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QRadioButton, QGroupBox,
                             QFileDialog, QComboBox, QMessageBox)

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

        # 4. 邮箱选择（新增）
        email_layout = QHBoxLayout()
        email_layout.addWidget(QLabel("报警邮箱:"))
        self.email_combo = QComboBox()
        # 添加三个管理员邮箱选项
        self.email_combo.addItem("管理员1 (1907872557@qq.com)", "1907872557@qq.com")
        self.email_combo.addItem("管理员2 (wqr20011989@163.com)", "wqr20011989@163.com")
        self.email_combo.addItem("管理员3 (903466339@qq.com)", "903466339@qq.com")
        # self.email_combo.addItem("管理员2 (Honglingxiang@kaifa.cn)", "Honglingxiang@kaifa.cn")
        # self.email_combo.addItem("管理员3 (XinHuZhang@kaifa.cn)", "XinHuZhang@kaifa.cn")
        # self.email_combo.addItem("管理员4 (ShaoHuawang1@kaifa.cn)", "ShaoHuawang1@kaifa.cn")
        # self.email_combo.addItem("管理员5 (xiaoyuzhong@kaifa.cn)", "xiaoyuzhong@kaifa.cn")



        email_layout.addWidget(self.email_combo)
        main_layout.addLayout(email_layout)

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
        # 新增：设置邮箱选择
        if hasattr(self.video_info, 'alert_email') and self.video_info.alert_email:
            # 查找对应的邮箱索引
            for index in range(self.email_combo.count()):
                if self.email_combo.itemData(index) == self.video_info.alert_email:
                    self.email_combo.setCurrentIndex(index)
                    break
            else:
                # 如果邮箱不在选项中，添加并选择
                self.email_combo.addItem(f"自定义: {self.video_info.alert_email}", self.video_info.alert_email)
                self.email_combo.setCurrentIndex(self.email_combo.count() - 1)

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

    """获取选择的邮箱"""
    def get_selected_email(self):
        selected_index = self.email_combo.currentIndex()
        if selected_index == -1:
            selected_index = 0
        return self.email_combo.itemData(selected_index)

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