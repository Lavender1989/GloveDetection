# email_sender.py
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import cv2
from datetime import datetime
import os


class EmailSender:
    def __init__(self):
        # 配置邮箱服务器
        self.smtp_server = "smtp.163.com"#这个是网易邮箱的SMTP服务器地址，如果用其他邮箱也可以
        self.smtp_port = 25
        self.sender_email = "19091601379@163.com"  # 发件人邮箱，这里是我自己的邮箱
        self.sender_password = "WTs5XEsp5BT68Rtb"  # 网易邮箱授权码,有效期180天，2025年8月21日开始生效，到期后需要更换

        # 三个管理员的邮箱地址（只有在这个列表里的邮箱地址才能收到邮件，需要增加的话，在下面新增就行）
        self.admin_emails = [
            # "1907872557@qq.com",
            "Honglingxiang@kaifa.cn",
            "XinHuZhang@kaifa.cn",
            "ShaoHuawang1@kaifa.cn",
            "xiaoyuzhong@kaifa.cn"
            # "wqr20011989@163.com",
            "903466339@qq.com"
        ]

    def send_alert_email(self, video_name, alert_message, alert_frame, selected_emails=None):
        """
        发送报警邮件

        Args:
            video_name: 视频源名称
            alert_message: 报警信息
            alert_frame: 报警帧（numpy数组）
            selected_emails: 选择的收件人列表，如果为None则发送给管理员列表里的第一个
        """
        try:
            # 如果没有指定收件人，使用所有管理员邮箱
            if selected_emails is None:
                recipients = self.admin_emails[0]
            else:
                recipients = selected_emails

            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = f"安全报警 - {video_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            # 邮件正文
            body = f"""
            手套佩戴监控检测系统报警通知

            报警时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            视频源: {video_name}
            报警信息: {alert_message}

            请及时处理！
            """
            msg.attach(MIMEText(body, 'plain'))

            # 添加报警帧图片
            if alert_frame is not None:
                # 将帧保存为临时图片文件
                temp_image_path = f"alert_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(temp_image_path, alert_frame)

                # 读取图片并附加到邮件
                with open(temp_image_path, 'rb') as f:
                    img_data = f.read()

                image = MIMEImage(img_data, name=os.path.basename(temp_image_path))
                msg.attach(image)

                # 删除临时文件
                os.remove(temp_image_path)

            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipients, msg.as_string())

            return True

        except Exception as e:
            print(f"发送邮件失败: {str(e)}")
            return False