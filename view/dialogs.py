import cv2
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QRadioButton, QGroupBox,
                             QFileDialog, QComboBox, QMessageBox, QListWidget,
                             QListWidgetItem, QAbstractItemView, QGridLayout, QCheckBox)
from model.db import Database

class EmailDialog(QDialog):
    """添加/编辑邮箱对话框"""
    
    def __init__(self, parent=None, email_id=None, name="", email=""):
        super().__init__(parent)
        self.setWindowTitle("添加新邮箱" if not email_id else "编辑邮箱")
        self.resize(300, 150)
        self.email_id = email_id
        
        # 初始化UI
        layout = QVBoxLayout()
        
        # 名称输入
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("名称:"))
        self.name_input = QLineEdit(name)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        # 邮箱输入
        email_layout = QHBoxLayout()
        email_layout.addWidget(QLabel("邮箱:"))
        self.email_input = QLineEdit(email)
        email_layout.addWidget(self.email_input)
        layout.addLayout(email_layout)
        
        # 底部按钮
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("确定")
        self.cancel_btn = QPushButton("取消")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def accept(self):
        """确认按钮处理"""
        # 简单的邮箱格式验证
        name = self.name_input.text().strip()
        email = self.email_input.text().strip()
        
        if not name or not email:
            QMessageBox.warning(self, "警告", "名称和邮箱不能为空")
            return
        
        # 简单的邮箱格式验证
        if '@' not in email:
            QMessageBox.warning(self, "警告", "请输入有效的邮箱地址")
            return
        
        super().accept()
    
    def get_email_info(self):
        """返回邮箱信息"""
        return {
            "id": self.email_id,
            "name": self.name_input.text().strip(),
            "email": self.email_input.text().strip()
        }


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
        self.db = Database()

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



        # 5. 邮箱选择（新增）支持多选
        email_group = QGroupBox("报警邮箱 (可多选)")
        email_group_layout = QVBoxLayout()
        
        # 添加全选复选框
        select_all_layout = QHBoxLayout()
        self.select_all_checkbox = QCheckBox("全选")
        # 使用更直接的clicked信号
        self.select_all_checkbox.clicked.connect(self.on_select_all_clicked)
        select_all_layout.addWidget(self.select_all_checkbox)
        select_all_layout.addStretch()
        email_group_layout.addLayout(select_all_layout)
        
        # 创建邮箱列表控件
        self.email_list = QListWidget()
        self.email_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        
        # 添加邮箱到列表
        self._load_emails()
        
        # 添加/删除邮箱按钮
        btn_layout = QHBoxLayout()
        self.add_email_btn = QPushButton("添加新邮箱")
        self.add_email_btn.clicked.connect(self._add_new_email)
        
        self.delete_email_btn = QPushButton("删除选中邮箱")
        self.delete_email_btn.clicked.connect(self._delete_email)
        
        btn_layout.addWidget(self.add_email_btn)
        btn_layout.addWidget(self.delete_email_btn)
        
        email_group_layout.addWidget(self.email_list)
        email_group_layout.addLayout(btn_layout)
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

    def on_select_all_clicked(self):
        """处理全选复选框点击事件"""
        # 使用checked属性来判断当前状态
        is_checked = self.select_all_checkbox.isChecked()
        for i in range(self.email_list.count()):
            item = self.email_list.item(i)
            item.setSelected(is_checked)
    
    def _load_emails(self):
        """从数据库加载邮箱列表"""
        self.email_list.clear()
        emails = self.db.get_all_emails()
        self.available_emails = [{"id": email.id, "name": email.name, "email": email.email} for email in emails]
        
        # 添加邮箱到列表
        for email_info in self.available_emails:
            item = QListWidgetItem(f"{email_info['name']} ({email_info['email']})")
            item.setData(Qt.ItemDataRole.UserRole, email_info['email'])
            item.setData(Qt.ItemDataRole.UserRole + 1, email_info['id'])  # 存储邮箱ID
            self.email_list.addItem(item)
    
    def _add_new_email(self):
        """添加新邮箱"""
        dialog = EmailDialog(self)
        if dialog.exec():
            email_info = dialog.get_email_info()
            
            # 添加到数据库
            if self.db.add_email(email_info['name'], email_info['email']):
                # 重新加载邮箱列表
                self._load_emails()
                # 选中新添加的邮箱
                for i in range(self.email_list.count()):
                    item = self.email_list.item(i)
                    if item.data(Qt.ItemDataRole.UserRole) == email_info['email']:
                        item.setSelected(True)
            else:
                QMessageBox.warning(self, "警告", "邮箱已存在或添加失败")
    
    def _delete_email(self):
        """删除选中的邮箱"""
        selected_items = self.email_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要删除的邮箱")
            return
        
        # 确认删除
        if QMessageBox.question(self, "确认删除", "确定要删除选中的邮箱吗？", 
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.No:
            return
        
        # 删除选中的邮箱
        for item in selected_items:
            email_id = item.data(Qt.ItemDataRole.UserRole + 1)
            if email_id:
                self.db.delete_email(email_id)
        
        # 重新加载邮箱列表
        self._load_emails()
        
        # 如果还有邮箱，默认选中第一个
        if self.email_list.count() > 0:
            self.email_list.setCurrentRow(0)
            self.email_list.item(0).setSelected(True)
    
    """获取选择的邮箱列表（多个邮箱用逗号分隔）"""
    def get_selected_email(self):
        selected_items = self.email_list.selectedItems()
        if not selected_items:
            # 如果没有选择，返回空字符串或第一个可用邮箱（如果有）
            if self.available_emails:
                return self.available_emails[0]['email']
            return ""
            
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
            "alert_email": self.get_selected_email(),  # 新增邮箱信息
            "detection_type": 1  # 默认检测类型，现在两个模型都会运行
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


class ModelSelectDialog(QDialog):
    """模型选择对话框"""
    log_message = pyqtSignal(str)  # 定义日志信号
    
    def __init__(self, parent=None, selected_models=None, model_confidence=None, model_thresholds=None):
        super().__init__(parent)
        self.setWindowTitle("模型选择")
        self.resize(400, 400)
        # 默认选择手套模型
        self.selected_models = selected_models or ['glove']
        # 使用集中配置的默认值
        self.model_confidence = model_confidence or DEFAULT_MODEL_CONFIDENCE
        # 添加报警阈值参数
        self.model_thresholds = model_thresholds or DEFAULT_MODEL_THRESHOLDS
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)

        # 模型选择组
        model_group = QGroupBox("请选择要使用的检测模型:")
        model_group_layout = QVBoxLayout()
        model_group.setLayout(model_group_layout)

        # 手套模型
        self.glove_checkbox = QCheckBox("手套检测 (检测未戴手套)")
        self.glove_checkbox.setChecked('glove' in self.selected_models)
        model_group_layout.addWidget(self.glove_checkbox)

        # 头部模型
        self.head_checkbox = QCheckBox("头部检测 (检测摸头违规行为)")
        self.head_checkbox.setChecked('head' in self.selected_models)
        model_group_layout.addWidget(self.head_checkbox)

        layout.addWidget(model_group)

        # 置信度设置
        confidence_group = QGroupBox("模型置信度设置:")
        confidence_group_layout = QGridLayout()
        confidence_group.setLayout(confidence_group_layout)

        # 手套模型置信度
        confidence_group_layout.addWidget(QLabel("手套检测置信度:"), 0, 0)
        self.glove_conf_input = QLineEdit(str(self.model_confidence.get('glove', 0.8)))
        self.glove_conf_input.setPlaceholderText("0.0-1.0")
        confidence_group_layout.addWidget(self.glove_conf_input, 0, 1)

        # 头部模型置信度
        confidence_group_layout.addWidget(QLabel("头部检测置信度:"), 1, 0)
        self.head_conf_input = QLineEdit(str(self.model_confidence.get('head', 0.8)))
        self.head_conf_input.setPlaceholderText("0.0-1.0")
        confidence_group_layout.addWidget(self.head_conf_input, 1, 1)

        layout.addWidget(confidence_group)

        # 报警阈值设置
        threshold_group = QGroupBox("报警阈值设置 (连续检测到危险的帧数):")
        threshold_group_layout = QGridLayout()
        threshold_group.setLayout(threshold_group_layout)

        # 手套模型阈值
        threshold_group_layout.addWidget(QLabel("手套检测报警阈值:"), 0, 0)
        self.glove_threshold_input = QLineEdit(str(self.model_thresholds.get('glove', 5)))
        self.glove_threshold_input.setPlaceholderText("1-20")
        threshold_group_layout.addWidget(self.glove_threshold_input, 0, 1)

        # 头部模型阈值
        threshold_group_layout.addWidget(QLabel("头部检测报警阈值:"), 1, 0)
        self.head_threshold_input = QLineEdit(str(self.model_thresholds.get('head', 3)))
        self.head_threshold_input.setPlaceholderText("1-20")
        threshold_group_layout.addWidget(self.head_threshold_input, 1, 1)

        layout.addWidget(threshold_group)

        # 按钮布局
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

    def get_selected_models(self):
        """获取选中的模型"""
        selected = []
        if self.glove_checkbox.isChecked():
            selected.append('glove')
        if self.head_checkbox.isChecked():
            selected.append('head')
        self.log_message.emit("selected_models: {}".format(selected))
        # 验证至少选择一个模型
        if not selected:
            QMessageBox.warning(self, "错误", "请至少选择一个检测模型")
            return None

        # 验证置信度输入
        try:
            glove_conf = float(self.glove_conf_input.text().strip())
            if not 0 <= glove_conf <= 1:
                QMessageBox.warning(self, "错误", "手套检测置信度必须在0.0-1.0之间")
                return None
        except ValueError:
            QMessageBox.warning(self, "错误", "手套检测置信度必须是有效数字")
            return None

        try:
            head_conf = float(self.head_conf_input.text().strip())
            if not 0 <= head_conf <= 1:
                QMessageBox.warning(self, "错误", "头部检测置信度必须在0.0-1.0之间")
                return None
        except ValueError:
            QMessageBox.warning(self, "错误", "头部检测置信度必须是有效数字")
            return None

        # 验证报警阈值
        try:
            glove_threshold = int(self.glove_threshold_input.text().strip())
            if not 1 <= glove_threshold <= 20:
                QMessageBox.warning(self, "错误", "手套检测报警阈值必须在1-20之间")
                return None
        except ValueError:
            QMessageBox.warning(self, "错误", "手套检测报警阈值必须是有效整数")
            return None

        try:
            head_threshold = int(self.head_threshold_input.text().strip())
            if not 1 <= head_threshold <= 20:
                QMessageBox.warning(self, "错误", "头部检测报警阈值必须在1-20之间")
                return None
        except ValueError:
            QMessageBox.warning(self, "错误", "头部检测报警阈值必须是有效整数")
            return None

        return {
            'models': selected,
            'glove_confidence': glove_conf,
            'head_confidence': head_conf,
            'glove_threshold': glove_threshold,
            'head_threshold': head_threshold
        }