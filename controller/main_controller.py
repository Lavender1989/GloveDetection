# view/main_controller.py
import os
import sys
import time
import numpy as np

import cv2
import torch
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer, pyqtSlot, QMetaObject, Q_ARG
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QImage
from PyQt6.QtWidgets import QTreeWidgetItem, QMessageBox

from model.db import Database, VideoSource
from view.dialogs import VideoSourceDialog, SceneDialog
from .video_capture_manager import VideoCaptureManager


class DetectionThread(QThread):
    """新版检测线程，对接新版 MultiDetectorWorker（内部含输入线程 + 推理线程）"""
    log_signal = pyqtSignal(str)
    alert_signal = pyqtSignal(str)
    frame_processed = pyqtSignal(int, QImage)
    rtsp_disconnected = pyqtSignal(int)

    def __init__(self, video_source, capture_manager):
        super().__init__()
        self.video_source = video_source
        self.capture_manager = capture_manager
        # self.running = True
        # self.paused = False
        self.detector = None
        
        self.enabled_models = {'glove': True, 'head': False}
        self.model_confidence = {'glove': 0.8, 'head': 0.8}
        self.model_thresholds = {'glove': 5, 'head': 10}

    def update_models(self, enabled, confidence, thresholds):
        # MainController动态更新模型配置
        self.enabled_models = enabled
        self.model_confidence = confidence
        self.model_thresholds = thresholds
        
        # 将更新后的配置传递给detector
        if hasattr(self, 'detector'):
            self.detector.update_models(enabled, confidence, thresholds)

    def run(self):
        from .worker import MultiDetectorWorker
        from .video_view_mapping import get_view_for_video, get_view_name

        video_name = self.video_source.name
        video_id = self.video_source.id
        video_url = self.video_source.path

        self.log_signal.emit(f"启动检测线程: {video_name}")

        # 视角推断
        view_index = get_view_for_video(video_url)
        print(f"[DEBUG] 视角推断: {view_index}")
        view_name = get_view_name(view_index)
        self.log_signal.emit(f"{video_name}: 加载{view_name}")

        # 模型配置
        models_config = {
            'glove': {
                'path': get_resource_path("../model/glove/best.pt"),
                'target_classes': ['bare'],
                'conf': self.model_confidence['glove'],
                'threshold': self.model_thresholds['glove'],
                'frame_threshold': 10,
                'trigger_mode': 'area',
                'enabled': self.enabled_models['glove']
            },
            'head': {
                'path': get_resource_path("../model/head/best.pt"),
                'target_classes': ['touch'],
                'conf': self.model_confidence['head'],
                'threshold': self.model_thresholds['head'],
                'frame_threshold': 10,
                'trigger_mode': 'any',
                'enabled': self.enabled_models['head']
            }
        }

        # 新版 MultiDetectorWorker：只做推理，不负责读帧
        self.detector = MultiDetectorWorker(
            models_config=models_config,
            video_name=video_name,
            view_index=view_index,
            alert_email="",
            capture_manager=self.capture_manager,
            video_id=video_id
        )

        # ⭐ 关键：把 worker 绑定到这个 QThread
        self.detector.moveToThread(self)

        # 日志 & 报警信号
        self.detector.log_message.connect(self.log_signal)
        self.detector.alert_message.connect(self.alert_signal)
        self.detector.proc_frame_ready.connect(
            lambda qimg: self.frame_processed.emit(video_id, qimg)
        )

        # 将视频源添加到 capture_manager
        ok = self.capture_manager.add_video_stream(
            video_id=video_id,
            video_url=video_url,
            fps_limit=30
        )

        if not ok:
            self.log_signal.emit(f"无法连接视频: {video_name}")
            self.rtsp_disconnected.emit(video_id)
            return

        # attach + start 必须在本线程事件循环中执行
        QMetaObject.invokeMethod(
            self.detector,
            "attach_video_source",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(object, self.capture_manager),
            Q_ARG(int, video_id)
        )

        QMetaObject.invokeMethod(
            self.detector,
            "start",
            Qt.ConnectionType.QueuedConnection
        )
        # 启动事件循环（非常重要）
        self.exec()

        # 线程退出时清理
        self.detector.stop()
        self.capture_manager.remove_video_stream(video_id)
        self.log_signal.emit(f"停止处理视频: {video_name}")

    @pyqtSlot()
    def pause(self):
        if self.detector:
            # 1️⃣ 停止推理
            QMetaObject.invokeMethod(
                self.detector, "pause", Qt.ConnectionType.QueuedConnection
            )
            # 2️⃣ 冻结视频
            self.capture_manager.pause_video(self.video_source.id)
            self.log_signal.emit(f"暂停处理视频: {self.video_source.name}")
    
    @pyqtSlot()
    def resume(self):
        if self.detector:
            # 1️⃣ 恢复视频
            self.capture_manager.resume_video(self.video_source.id)
            # 2️⃣ 恢复推理
            QMetaObject.invokeMethod(
                self.detector, "resume", Qt.ConnectionType.QueuedConnection
            )
        # self.paused = False
            self.log_signal.emit(f"继续处理视频: {self.video_source.name}")

    def stop(self):
        if self.detector:
            QMetaObject.invokeMethod(
                self.detector,
                "stop",
                Qt.ConnectionType.QueuedConnection
            )
        # self.running = False
        # self.paused = False
        self.quit()
        self.wait()

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

class MainController(QObject):
    # 新增信号
    video_frame_updated = pyqtSignal(int, QImage)  # 视频帧更新信号 (video_id, qimage)
    video_added = pyqtSignal(int, str)  # 新增视频信号 (video_id, name)
    video_removed = pyqtSignal(int)  # 移除视频信号 (video_id)

    def __init__(self, main_window):
        super().__init__()  # 初始化QObject
        self.main_window = main_window
        self.db = Database()
        self.current_scene_id = None
        # 管理所有DetectionThread
        self.detection_threads =  {}  # 改为字典存储 {video_id: DetectionThread}
        
        # 创建全局视频捕获管理器（不传URL）
        self.video_capture_manager = VideoCaptureManager()
        self.video_capture_manager.log_message.connect(self.log)
        self.video_capture_manager.rtsp_disconnected.connect(self.handle_rtsp_disconnect)
        
        # 模型路径配置
        self.model_paths = {
            'glove': get_resource_path("../model/glove/best.pt"),  # 手套模型
            'head': get_resource_path("../model/head/best.pt")    # 头部模型
        }
        # 启用的模型
        self.enabled_models = {'glove': True,'head': False}
        # 模型置信度
        self.model_confidence = {'glove': 0.8,'head': 0.8}
        # 模型报警阈值（连续检测到危险的帧数）
        self.model_thresholds = {'glove': 5,'head': 10}

        # 初始化日志模型
        self.log_model = QStandardItemModel()
        self.main_window.log_box.setModel(self.log_model)

        # 初始化UI和信号连接
        self.init_ui()
        # 连接信号与槽函数
        self.init_signals()
        self.video_added.connect(main_window.add_video_tab)
        self.video_removed.connect(main_window.remove_video_tab)
        self.video_frame_updated.connect(main_window.update_video_frame)
        # 将控制器设置到窗口
        self.main_window.controller = self

    def init_ui(self):
        """初始化UI数据"""
        # 加载所有场景到下拉框
        self.load_scenes_to_combobox()

        # 修改：设置列数和表头（增加邮箱列）
        self.main_window.video_list.setColumnCount(5)  # 改为5列
        self.main_window.video_list.setHeaderLabels(["选择", "名称", "路径", "类型", "报警邮箱"])

        # 添加开启检测按钮（如果UI中没有）
        self.main_window.start_detection_btn.clicked.connect(self.start_detection)

    def init_signals(self):
        """连接信号与槽函数"""
        # 场景相关
        self.main_window.choose_scene.currentIndexChanged.connect(self.on_scene_changed)
        self.main_window.add_scene_btn.clicked.connect(self.add_scene)
        self.main_window.delete_scene_btn.clicked.connect(self.delete_current_scene)

        # 视频源相关
        self.main_window.add_video_btn.clicked.connect(self.add_video_source)
        self.main_window.delete_video_btn.clicked.connect(self.delete_video_source)
        self.main_window.edit_video_btn.clicked.connect(self.edit_video_source)

        # 添加停止检测按钮连接
        # self.main_window.close_detection_btn.clicked.connect(self.stop_all_detections)
        self.main_window.close_detection_btn.clicked.connect(self.pause_all_detection)
        # 视频列表项点击事件（处理选择状态）
        self.main_window.video_list.itemChanged.connect(self.on_video_item_changed)

        # 双击编辑
        self.main_window.video_list.itemDoubleClicked.connect(self.on_video_item_double_clicked)

    def load_scenes_to_combobox(self):
        """加载所有场景到下拉框"""
        self.main_window.choose_scene.clear()
        scenes = self.db.get_all_scenes()

        for scene in scenes:
            self.main_window.choose_scene.addItem(scene.name, scene.id)

        # 如果有场景，默认选择第一个
        if scenes:
            self.current_scene_id = scenes[0].id
            self.load_videos_for_current_scene()

    def on_scene_changed(self, index):
        """切换场景时加载对应的视频源"""
        if index >= 0:
            self.current_scene_id = self.main_window.choose_scene.itemData(index)
            self.load_videos_for_current_scene()

    def load_videos_for_current_scene(self):
        """加载当前场景下的所有视频源"""
        self.main_window.video_list.clear()
        if not self.current_scene_id:
            return

        videos = self.db.get_videos_by_scene(self.current_scene_id)
        for video in videos:
            item = QTreeWidgetItem()
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)

            #控制复选框选中状态（基于is_true）
            check_state = Qt.CheckState.Checked if video.is_true else Qt.CheckState.Unchecked
            item.setCheckState(0, check_state)

            item.setText(1, video.name)
            item.setText(2, video.path)
            # 设置视频类型文本
            type_text = "本地文件" if video.type == 1 else \
                "RTSP地址" if video.type == 2 else "本机摄像头"
            item.setText(3, type_text)

            # 设置报警邮箱信息
            if not video.alert_email or video.alert_email == "all":
                email_text = "所有管理员"
            else:
                # 处理多个邮箱的显示
                emails = [email.strip() for email in video.alert_email.split(',')]
                if len(emails) > 2:
                    # 如果邮箱数量超过2个，只显示前两个并显示总数
                    email_text = f"{emails[0]}, {emails[1]} 等共{len(emails)}个"
                else:
                    # 直接显示所有邮箱
                    email_text = ", ".join(emails)
            item.setText(4, email_text)

            # 存储视频ID，方便后续操作
            item.setData(0, Qt.ItemDataRole.UserRole, video.id)
            item.setData(1, Qt.ItemDataRole.UserRole, video.type)

            self.main_window.video_list.addTopLevelItem(item)

    def add_scene(self):
        """添加新场景"""
        dialog = SceneDialog(self.main_window)
        if dialog.exec():
            scene_name = dialog.get_scene_name()
            if scene_name:
                if self.db.add_scene(scene_name):
                    self.log(f"添加场景成功: {scene_name}")
                    self.load_scenes_to_combobox()
                else:
                    QMessageBox.warning(self.main_window, "错误", "场景名称已存在")

    def set_selected_models(self, selected_models):
        """设置选中的模型
        
        Args:
            selected_models: 包含选中模型信息的字典
        """
        try:
            # 更新启用的模型状态
            self.enabled_models = {
                'glove': 'glove' in selected_models['models'],
                'head': 'head' in selected_models['models']
            }
            
            # 更新置信度
            self.model_confidence = {
                'glove': selected_models['glove_confidence'],
                'head': selected_models['head_confidence']
            }
            
            # 更新报警阈值
            self.model_thresholds = {
                'glove': selected_models['glove_threshold'],
                'head': selected_models['head_threshold']
            }
            
            self.log(f"更新模型选择: {selected_models}")
            # 添加打印配置信息
            print(f"[模型配置] 启用模型: {self.enabled_models}")
            print(f"[模型配置] 置信度设置: {self.model_confidence}")
            
            # 如果有正在运行的检测线程，更新它们的模型设置
            for video_id, thread in self.detection_threads.items():
                if thread and thread.isRunning():
                    # 在DetectionThread类中实现
                    thread.update_models(self.enabled_models, self.model_confidence, self.model_thresholds)
            
            QMessageBox.information(self.main_window, "成功", "模型选择已更新")
            
        except Exception as e:
            self.log(f"[错误] 更新模型选择失败: {str(e)}")
            QMessageBox.critical(self.main_window, "错误", f"更新模型选择失败: {str(e)}")

    def delete_current_scene(self):
        """删除当前场景"""
        if not self.current_scene_id:
            return

        scene_name = self.main_window.choose_scene.currentText()
        reply = QMessageBox.question(
            self.main_window, "确认删除",
            f"确定要删除场景 '{scene_name}' 吗？\n相关视频源也将被删除。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.db.delete_scene(self.current_scene_id):
                self.log(f"删除场景成功: {scene_name}")
                self.load_scenes_to_combobox()
            else:
                QMessageBox.warning(self.main_window, "错误", "删除场景失败")

    def add_video_source(self):
        """添加视频源"""
        if not self.current_scene_id:
            QMessageBox.warning(self.main_window, "警告", "请先选择一个场景")
            return

        dialog = VideoSourceDialog(self.main_window, scene_id=self.current_scene_id)
        if dialog.exec():
            try:
                video_info = dialog.get_video_info()
                if not video_info["name"] or not video_info["path"]:
                    QMessageBox.warning(self.main_window, "警告", "名称和路径不能为空")
                    return
                # 创建VideoSource对象并保存到数据库
                video = VideoSource(
                    id=0,  # 数据库会自动生成ID
                    name=video_info["name"],
                    path=video_info["path"],
                    is_true=video_info["is_true"],
                    scene_id=video_info["scene_id"],
                    type=video_info["type"],
                    is_valid=True,  # 新增：设置有效性
                    alert_email = video_info["alert_email"],  # 新增：添加报警邮箱
                    detection_type = 1  # 默认检测类型，现在两个模型都会运行
                )

                video_id = self.db.add_video_source(video)
                if video_id:
                    self.log(f"添加视频源成功: {video.name}")
                    self.load_videos_for_current_scene()
            except Exception as e:
                self.log(f"添加视频源失败: {str(e)}")
                QMessageBox.critical(self.main_window, "错误", f"添加过程出错: {str(e)}")

    def delete_video_source(self):
        """删除选中的视频源"""
        selected_items = self.main_window.video_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self.main_window, "警告", "请先选择要删除的视频源")
            return

        item = selected_items[0]
        video_id = item.data(0, Qt.ItemDataRole.UserRole)
        video_name = item.text(1)

        reply = QMessageBox.question(
            self.main_window, "确认删除",
            f"确定要删除视频源 '{video_name}' 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.db.delete_video_source(video_id):
                self.log(f"删除视频源成功: {video_name}")
                self.load_videos_for_current_scene()
            else:
                QMessageBox.warning(self.main_window, "错误", "删除视频源失败")

    """编辑选中的视频源"""
    def edit_video_source(self):
        selected_items = self.main_window.video_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self.main_window, "警告", "请先选择要编辑的视频源")
            return

        item = selected_items[0]
        video_id = item.data(0, Qt.ItemDataRole.UserRole)

        # 获取当前视频源信息
        videos = self.db.get_videos_by_scene(self.current_scene_id)
        video_info = next((v for v in videos if v.id == video_id), None)

        if video_info:
            dialog = VideoSourceDialog(self.main_window, video_info, self.current_scene_id)
            if dialog.exec():
                updated_info = dialog.get_video_info()

                if not updated_info["name"] or not updated_info["path"]:
                    QMessageBox.warning(self.main_window, "警告", "名称和路径不能为空")
                    return

                # 更新视频源信息
                updated_video = VideoSource(
                    id=video_info.id,
                    name=updated_info["name"],
                    path=updated_info["path"],
                    is_true=video_info.is_true,
                    is_valid=True,  # 更新连接状态
                    scene_id=self.current_scene_id,
                    type=updated_info["type"],
                    alert_email=updated_info["alert_email"],
                    detection_type=1  # 默认检测类型，现在两个模型都会运行
                )

                if self.db.update_video_source(updated_video):
                    self.log(f"更新视频源成功: {updated_video.name}")
                    self.load_videos_for_current_scene()

    def on_video_item_changed(self, item, column):
        """处理视频项选择状态变化"""
        if column == 0:  # 只处理第一列（选择框）的变化
            video_id = item.data(0, Qt.ItemDataRole.UserRole)
            is_checked = item.checkState(0) == Qt.CheckState.Checked
            self.db.update_video_selection(video_id, is_checked)
            video_name = item.text(1)
            self.log(f"{'选中' if is_checked else '取消选中'} 视频源: {video_name}")

    def on_video_item_double_clicked(self, item, column):
        """双击编辑视频源"""
        self.edit_video_source()

    def start_detection(self):
        """开始或继续检测选中的视频源"""
        try:
            # 输出当前模型配置
            print(f"[模型配置] 启用模型: {self.enabled_models}")
            print(f"[模型配置] 置信度设置: {self.model_confidence}")
            
            # 获取当前场景下所有选中的视频
            videos = [v for v in self.db.get_videos_by_scene(self.current_scene_id)
                  if v.is_true]

            if not videos:
                QMessageBox.warning(self.main_window, "警告", "请先选择要检测的视频源")
                return

            self.log(f"开始对 {len(videos)} 个视频源进行检测...")
            
            # 为每个选中的视频源创建或恢复检测线程
            for video in videos:
                try:
                    # 检查是否已有该视频的线程
                    if video.id in self.detection_threads:
                        self.log(f"视频已在运行: {video.name}")
                        # thread = self.detection_threads[video.id]
                        # if thread.paused:
                        #     thread.resume()
                        self.detection_threads[video.id].resume()
                    else:
                        # 创建新线程，传递VideoCaptureManager
                        thread = DetectionThread(video, self.video_capture_manager)
                        # 更新模型配置
                        thread.enabled_models = self.enabled_models
                        thread.model_confidence = self.model_confidence
                        thread.model_thresholds = self.model_thresholds
                        thread.log_signal.connect(self.log)
                        thread.alert_signal.connect(lambda msg, vid=video.name:
                                                    self.log(f"[报警] {vid}: {msg}"))

                        # 在创建线程时就连接上新增的RTSP断流信号
                        if video.type == 2:  # 只对RTSP类型生效
                            thread.rtsp_disconnected.connect(self.handle_rtsp_disconnect)

                        # 连接帧处理完成信号
                        thread.frame_processed.connect(self.on_frame_processed)
                        self.detection_threads[video.id] = thread

                        # 启动线程前进行额外检查
                        if not hasattr(thread, 'video_source'):
                            self.log(f"线程初始化失败: {video.name}")
                            continue

                        thread.start()
                        # 通知UI添加视频标签页
                        self.video_added.emit(video.id, video.name)

                except Exception as e:
                    self.log(f"创建检测线程失败 {video.name}: {str(e)}")
                    import traceback
                    self.log(f"详细错误: {traceback.format_exc()}")

        except Exception as e:
            self.log(f"start_detection 方法出错: {str(e)}")
            import traceback
            self.log(f"详细堆栈: {traceback.format_exc()}")
            QMessageBox.critical(self.main_window, "错误", f"启动检测失败: {str(e)}")


    def handle_rtsp_disconnect(self, video_id):
        """处理RTSP断流：通知UI并触发重连"""
        self.log(f"RTSP断流: 视频源ID={video_id}，将自动重连")
        # 修复：删除多余的self参数，2秒后触发重连
        QTimer.singleShot(2000, lambda: self.restart_rtsp_detection(video_id))

    def restart_rtsp_detection(self, video_id):
        """
        重构重连流程：
        1. 优先用线程里缓存的 video_source（避免重复查库）
        2. 查库失败时，明确提示“数据库查询异常”而非“未找到视频源”
        3. 简化流程：直接复用原线程的 video_source 重连
        """
        # 1. 从线程字典拿旧线程（优先用缓存的 video_source）
        if video_id in self.detection_threads:
            thread = self.detection_threads[video_id]
            video = thread.video_source  # 直接用线程里的 video_source
            if not video:
                self.log(f"重连警告: 线程 {video_id} 的 video_source 为空，尝试查库补救")
                video = self.db.get_video_by_id(video_id)  # 兜底查库

        # 2. 查库兜底（如果线程里没有，或查库也失败）
        if not video:
            video = self.db.get_video_by_id(video_id)
            if not video:
                self.log(f"重连失败: 数据库中也未找到视频源ID={video_id}！请检查配置")
                return  # 确实找不到，无法重连

        # 3. 验证视频源状态（必须是 RTSP 且已选中）
        if video.type != 2:
            self.log(f"重连跳过: 视频 {video.name} 不是 RTSP 类型（类型={video.type}）")
            return
        if not video.is_true:
            self.log(f"重连跳过: 视频 {video.name} 未选中检测（is_true={video.is_true}）")
            return

        # 4. 停止旧线程（如果存在）
        self.stop_video_detection(video_id)  # 封装成通用方法，停止并清理线程

        # 5. 创建新线程并重连
        try:
            new_thread = DetectionThread(video, self.video_capture_manager)
            # 重新连接信号
            new_thread.log_signal.connect(self.log)
            new_thread.alert_signal.connect(lambda msg, vid=video.name: self.log(f"[报警] {vid}: {msg}"))
            new_thread.rtsp_disconnected.connect(self.handle_rtsp_disconnect)
            new_thread.frame_processed.connect(self.on_frame_processed)

            self.detection_threads[video_id] = new_thread
            new_thread.start()
            self.log(f"RTSP重连成功: {video.name}(ID={video_id})已重启检测")
            self.video_added.emit(video_id, video.name)  # 恢复标签页
        except Exception as e:
            self.log(f"重连线程创建失败: {video.name} - {str(e)}")
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")
    """暂停所有检测线程"""
    def pause_all_detection(self):
        if not self.detection_threads:
            QMessageBox.information(self.main_window, "提示", "没有正在运行的检测线程")
            return

        for thread in self.detection_threads.values():
            # if thread.isRunning() and not thread.paused:
            if thread.isRunning():  # UI层不知道paused的内部状态，只发pause
                thread.pause()
        self.log("已暂停所有检测线程")

    def resume_all_detection(self):
        if not self.detection_threads:
            QMessageBox.information(self.main_window, "提示", "没有正在运行的检测线程")
            return

        for thread in self.detection_threads.values():
            if thread.isRunning():  # UI层不知道paused的内部状态，只发resume
                thread.resume()
        self.log("已恢复所有检测线程")
    

    """停止所有检测线程 不再【停止检测】调用"""
    def stop_all_detections(self):
        for video_id in list(self.detection_threads.keys()):
            self.stop_video_detection(video_id)
        self.detection_threads.clear()

    """停止指定视频的检测"""
    def stop_video_detection(self, video_id):
        if video_id not in self.detection_threads:
            return
        thread = self.detection_threads[video_id]
        if thread.isRunning():
            thread.stop()  # 等待线程完全停止            
        del self.detection_threads[video_id]
        self.video_removed.emit(video_id)
        self.log(f"已停止视频源 {video_id} 的检测线程")

    """处理检测线程发送的处理后帧"""
    def on_frame_processed(self, video_id, qimage):
        # print(f"[QIMAGE] width={qimage.width()}, height={qimage.height()}, bytesPerLine={qimage.bytesPerLine()}")
        # print(f"[DEBUG] on_frame_processed: 收到视频ID {video_id} 的帧，尺寸: {qimage.width()}x{qimage.height()}")
        self.video_frame_updated.emit(video_id, qimage)

    def log(self, message):
        """添加日志信息"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        item = QStandardItem(log_message)

        # 如果是报警信息，设置为红色
        if "[报警]" in log_message:
            item.setForeground(QtGui.QColor("red"))

        self.log_model.appendRow(item)
        # 自动滚动到底部
        self.main_window.log_box.scrollToBottom()
    # def log(self, message, is_success=False, is_error=False):
    #     timestamp = time.strftime("%H:%M:%S")
    #     log_message = f"[{timestamp}] {message}"
    #
    #     item = QStandardItem(log_message)
    #     # 日志分类染色
    #     if is_success:
    #         item.setForeground(QtGui.QColor("green"))
    #     elif is_error:
    #         item.setForeground(QtGui.QColor("red"))
    #     elif "[报警]" in log_message:
    #         item.setForeground(QtGui.QColor("red"))
    #
    #     self.log_model.appendRow(item)
    #     self.main_window.log_box.scrollToBottom()

    def cleanup(self):
        self.stop_all_detections()
        if hasattr(self, 'db'):
            self.db.close()