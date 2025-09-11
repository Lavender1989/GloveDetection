# main.py
运行主程序，开启检测，可以选择本地视频或者rtsp链接检测

# area
存放的是不同视角下待检测区域的标注文件

# controller
- detector_worker.py 是最近一版的检测器，其他后缀都是以前的检测器
- main_controller.py 处理线程
- video_view_mapping.py 定义不同视角(监控，本地视频对应哪个检测区域,)，每次采集新的视频数据可以在这里增加
-test_video_view_mapping.py 主要是测试视角关系对应是否正确
-VideoDetector.py 是在detect_video中测试好的检测器的封装(no use)

# model
- best.pt 训练好的yolov8模型权重(每次替换)
- db.py 数据库
- email_sender.py 更新警报邮件接收人，修改后还需要再view/dialogs.py下修改

# view
界面相关代码


