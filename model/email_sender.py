# email_sender.py
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
import cv2
from datetime import datetime
import os


class EmailSender:
    def __init__(self):
        # 配置邮箱服务器
        self.smtp_server = "smtp.163.com"#这个是网易邮箱的SMTP服务器地址，如果用其他邮箱也可以
        # self.smtp_port = 25
        self.smtp_port = 465  # 网易邮箱推荐使用465端口进行SSL加密连接
        self.sender_email = "wqr20011989@163.com"  # 发件人邮箱，这里是我自己的邮箱
        self.sender_password = "PCvs53uSKA8cP2xg"  # PCvs53uSKA8cP2xg 网易邮箱授权码,有效期180天，2025年12月1日开始生效，到期后需要更换

        # 管理员的邮箱地址（只有在这个列表里的邮箱地址才能收到邮件，需要增加的话，在下面新增就行）
        self.admin_emails = [
            # "1907872557@qq.com",
            "903466339@qq.com",
            "Honglingxiang@kaifa.cn",
            "xiaoyuzhong@kaifa.cn",
            "453851508@qq.com",
            "pqashifta@kaifa.cn",
            "xuhongdong@kaifa.cn",
            "jingming1@kaifa.cn"
        ]

    def send_alert_email(self, video_name, alert_message, alert_frame, selected_emails=None, original_frame=None, video_path=None):
        """
        发送报警邮件，支持发送报警图片和视频

        Args:
            video_name: 视频源名称
            alert_message: 报警信息
            alert_frame: 报警帧（numpy数组）
            selected_emails: 选择的收件人，支持单个邮箱字符串或多个邮箱用逗号分隔的字符串
            original_frame: 原始帧（numpy数组，无标注，可选）
            video_path: 视频文件路径（可选，包含报警前后的视频片段）
        """
        try:
            # 处理收件人列表
            if selected_emails is None or selected_emails == "":
                # 如果没有指定收件人或收件人为空字符串，使用所有管理员邮箱
                recipients = self.admin_emails
            else:
                # 处理逗号分隔的多个邮箱
                if isinstance(selected_emails, str):
                    # 分割并清理邮箱列表
                    recipients = [email.strip() for email in selected_emails.split(',') if email.strip()]
                else:
                    recipients = [selected_emails]  # 确保是列表格式

            # 确保有收件人
            if not recipients:
                print("没有有效的收件人邮箱")
                return False

            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = f"安全报警 - {video_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            # 构建邮件正文
            body_parts = [
                "手套佩戴监控检测系统报警通知",
                "",
                f"报警时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"视频源: {video_name}",
                f"报警信息: {alert_message}",
                ""
            ]
            
            # 根据提供的附件类型添加说明
            has_images = original_frame is not None
            has_video = video_path is not None and os.path.exists(video_path)
            
            if has_images and has_video:
                body_parts.append("请查看附件中的报警图片（有标注和无标注版本）和视频片段。")
            elif has_images:
                body_parts.append("请查看附件中的报警图片（有标注和无标注版本）。")
            elif has_video:
                body_parts.append("请查看附件中的报警视频片段。")
            else:
                body_parts.append("请及时处理！")
            
            body = "\n".join(body_parts)
            msg.attach(MIMEText(body, 'plain'))

            # 添加报警帧图片（带标注）
            if alert_frame is not None:
                # 将帧保存为临时图片文件
                temp_annotated_path = f"alert_frame_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(temp_annotated_path, alert_frame)

                # 读取图片并附加到邮件
                with open(temp_annotated_path, 'rb') as f:
                    img_data = f.read()

                image = MIMEImage(img_data, name="带标注报警图片.jpg")
                msg.attach(image)

                # 删除临时文件
                os.remove(temp_annotated_path)

            # 添加原始帧图片（无标注，如果提供）
            if original_frame is not None:
                # 将原始帧保存为临时图片文件
                temp_original_path = f"alert_frame_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(temp_original_path, original_frame)

                # 读取图片并附加到邮件
                with open(temp_original_path, 'rb') as f:
                    original_img_data = f.read()

                original_image = MIMEImage(original_img_data, name="无标注原始图片.jpg")
                msg.attach(original_image)

                # 删除临时文件
                os.remove(temp_original_path)
            
            # 添加视频附件（如果提供且文件存在）
            if video_path and os.path.exists(video_path):
                try:
                    file_size = os.path.getsize(video_path)
                    print(f"视频文件大小: {file_size} 字节")
                    
                    with open(video_path, 'rb') as f:
                        video_data = f.read()
                        print(f"读取的视频数据大小: {len(video_data)} 字节")
                        
                        # 设置正确的 MIME 类型为 video/mp4
                        video_attachment = MIMEApplication(video_data, _subtype='mp4')
                        video_attachment.add_header('Content-Disposition', 'attachment', 
                                                  filename=f"报警视频_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                        video_attachment.add_header('Content-Type', 'video/mp4')
                        msg.attach(video_attachment)
                    print(f"视频附件已添加: {video_path}")
                except Exception as e:
                    print(f"添加视频附件时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # 发送邮件 - 使用SSL连接（适用于465端口）
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                # 对于SSL连接，不需要调用starttls
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipients, msg.as_string())

            print(f"成功发送邮件到 {len(recipients)} 个收件人")
            print(f"- 附件信息：带标注图片{'、无标注图片' if original_frame is not None else ''}{'、视频片段' if video_path and os.path.exists(video_path) else ''}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            print(f"SMTP认证失败: {str(e)} - 请检查邮箱地址和授权码是否正确")
            return False
        except smtplib.SMTPException as e:
            print(f"SMTP服务器错误: {str(e)} - 请检查SMTP配置和网络连接")
            return False
        except Exception as e:
            print(f"发送邮件失败: {str(e)}")
            return False