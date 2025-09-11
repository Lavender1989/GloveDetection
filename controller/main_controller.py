# view/main_controller.py
import os
import sys
import time

import cv2
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QImage
from PyQt6.QtWidgets import QTreeWidgetItem, QMessageBox

from model.db import Database, VideoSource
from view.dialogs import VideoSourceDialog, SceneDialog


class DetectionThread(QThread):
    """视频检测线程（支持暂停/继续）"""
    log_signal = pyqtSignal(str)
    alert_signal = pyqtSignal(str)
    frame_processed = pyqtSignal(int, QImage)  # 新增信号：帧处理完成
    rtsp_disconnected = pyqtSignal(int)  # 新增：RTSP断流信号，携带video_id

    def __init__(self, video_source, model_path, interval):
        super().__init__()
        self.video_source = video_source
        self.model_path = model_path
        self.running = False  # 线程是否运行
        self.paused = False   # 线程是否暂停
        self.detector = None
        self.show_ui = True
        self.interval = interval
        self.cap = None       # 保存视频捕获对象，用于暂停后继续
        self.frame_pos = 0    # 记录当前帧位置（用于文件视频）

    def run(self):
        self.running = True
        self.paused = False
        self.log_signal.emit(f"开始处理视频: {self.video_source.name}")

        try:
            from .detector_worker import DetectorWorker
            from .video_view_mapping import get_view_for_video, get_view_name
            
            # 日志输出视频类型
            video_type = "RTSP地址" if self.video_source.path.lower().startswith("rtsp://") else "本地文件"
            self.log_signal.emit(f"视频类型: {video_type}")
            
            # 提前确定视角并输出日志
            view_index = get_view_for_video(self.video_source.path)
            view_name = get_view_name(view_index)
            self.log_signal.emit(f"{self.video_source.name}: 成功加载{view_name}")
            
            self.detector = DetectorWorker(self.model_path,
                                           self.video_source.name,  # 或改为self.video_source.path
                                           view_index,  # 直接传递已计算好的视角索引
                                           self.video_source.alert_email)# 新增：报警邮箱
            self.detector.log_message.connect(self.log_signal)
            self.detector.alert_message.connect(self.alert_signal)

            # 连接检测器的帧处理完成信号
            self.detector.proc_frame_ready.connect(
                lambda img: self.frame_processed.emit(self.video_source.id, img)
            )

            # 视频处理主循环（RTSP断流时不退出，循环重连）
            while self.running:
                # 暂停逻辑：暂停时休眠，不占用CPU
                while self.paused and self.running:
                    self.msleep(100)
                    continue

                # 打开视频源（如果是首次运行或视频已关闭）
                if not self.cap or not self.cap.isOpened():
                    self.log_signal.emit(f"尝试连接视频源: {self.video_source.name}")
                    self.cap = cv2.VideoCapture(self.video_source.path)
                    # RTSP连接需要时间，等待1秒确认是否成功
                    time.sleep(1)

                    if not self.cap.isOpened():
                        self.log_signal.emit(f"连接失败，3秒后重试: {self.video_source.name}")
                        self.rtsp_disconnected.emit(self.video_source.id)  # 发送断流通知
                        self.msleep(3000)  # 3秒后再重试，避免频繁重试
                        continue  # 不退出循环，继续尝试重连

                # 2. 连接成功后，读取帧并处理
                ret, frame = self.cap.read()
                if not ret:
                    self.log_signal.emit(f"帧读取失败，尝试重连: {self.video_source.name}")
                    self.rtsp_disconnected.emit(self.video_source.id)
                    self.cap.release()  # 释放无效连接
                    self.cap = None  # 标记为未连接，触发下一轮重连
                    self.msleep(2000)
                    continue  # 不退出循环，继续重连

                # 3. 按间隔处理帧（避免每帧都处理，降低CPU占用）
                self.frame_count = getattr(self, 'frame_count', 0) + 1
                if self.frame_count % self.interval == 0:
                    try:
                        self.detector.process_frame(frame)
                    except Exception as e:
                        self.log_signal.emit(f"帧处理错误: {str(e)}")
                        import traceback
                        self.log_signal.emit(f"错误详情: {traceback.format_exc()}")

                # 4. 控制帧率（避免读取过快）
                time.sleep(0.01)

        except Exception as e:
            self.log_signal.emit(f"线程异常: {self.video_source.name} - {str(e)}")
            import traceback
            self.log_signal.emit(f"异常详情: {traceback.format_exc()}")
        finally:
            # 只有线程完全停止时，才释放资源
            if not self.running and self.cap:
                self.cap.release()
                self.cap = None
            # self.log_signal.emit(f"停止处理视频: {self.video_source.name}")


    def pause(self):
        """暂停线程"""
        self.paused = True
        self.log_signal.emit(f"暂停处理视频: {self.video_source.name}")

    def resume(self):
        """继续线程"""
        self.paused = False
        self.log_signal.emit(f"继续处理视频: {self.video_source.name}")

    def stop(self):
        """完全停止线程（释放资源）"""
        self.running = False
        self.paused = False
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
        self.detection_threads =  {}  # 改为字典存储 {video_id: DetectionThread}
        self.model_path = get_resource_path("../model/best.pt")

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
        self.main_window.close_detection_btn.clicked.connect(self.pause_detection)

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

            # 新增：设置报警邮箱信息
            email_text = "所有管理员" if not video.alert_email or video.alert_email == "all" else video.alert_email
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
                    alert_email = video_info["alert_email"]  # 新增：添加报警邮箱
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
                    alert_email=updated_info["alert_email"]
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
            # 获取当前场景下所有选中的视频
            selected_videos = []
            videos = self.db.get_videos_by_scene(self.current_scene_id)
            for video in videos:
                if video.is_true:
                    selected_videos.append(video)

            if not selected_videos:
                QMessageBox.warning(self.main_window, "警告", "请先选择要检测的视频源")
                return

            self.log(f"开始对 {len(selected_videos)} 个视频源进行检测...")
            frame_interval = min(5, max(3, len(selected_videos) // 2))

            # 为每个选中的视频源创建或恢复检测线程
            for video in selected_videos:
                try:
                    # 检查是否已有该视频的线程
                    if video.id in self.detection_threads:
                        thread = self.detection_threads[video.id]
                        if thread.paused:
                            thread.resume()
                    else:
                        # 创建新线程
                        thread = DetectionThread(video, self.model_path, frame_interval)
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
            frame_interval = min(5, max(3, len(self.db.get_videos_by_scene(self.current_scene_id)) // 2))
            new_thread = DetectionThread(video, self.model_path, frame_interval)
            # 重新连接信号
            new_thread.log_signal.connect(self.log)
            new_thread.alert_signal.connect(lambda msg, vid=video.name: self.log(f"[报警] {vid}: {msg}"))
            new_thread.rtsp_disconnected.connect(self.handle_rtsp_disconnect)
            new_thread.frame_processed.connect(self.on_frame_processed)

            self.detection_threads[video_id] = new_thread
            new_thread.start()
            self.log(f"RTSP重连成功: {video.name}（ID={video_id}）已重启检测")
            self.video_added.emit(video_id, video.name)  # 恢复标签页
        except Exception as e:
            self.log(f"重连线程创建失败: {video.name} - {str(e)}")
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")
    """暂停所有检测线程"""
    def pause_detection(self):
        if not self.detection_threads:
            QMessageBox.information(self.main_window, "提示", "没有正在运行的检测线程")
            return

        for thread in self.detection_threads.values():
            if thread.isRunning() and not thread.paused:
                thread.pause()
        self.log("已暂停所有检测线程")

    """停止所有检测线程"""
    def stop_all_detections(self):
        for video_id in list(self.detection_threads.keys()):
            self.stop_video_detection(video_id)
        self.detection_threads.clear()

    """停止指定视频的检测"""
    def stop_video_detection(self, video_id):
        if video_id in self.detection_threads:
            thread = self.detection_threads[video_id]
            if thread.isRunning():
                thread.stop()  # 等待线程完全停止
            del self.detection_threads[video_id]
            self.video_removed.emit(video_id)
            self.log(f"已停止视频源 {video_id} 的检测线程")

    """处理检测线程发送的处理后帧"""
    def on_frame_processed(self, video_id, qimage):
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