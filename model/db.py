# model/db.py
import os
import sqlite3
from dataclasses import dataclass
from typing import List


@dataclass
class Scene:
    id: int
    name: str


@dataclass
class Email:
    id: int
    name: str
    email: str


@dataclass
class VideoSource:
    id: int
    name: str
    path: str
    is_true: bool
    is_valid: bool  # 是否可连接（RTSP专用）
    scene_id: int
    type: int  # 1:本地文件, 2:RTSP, 3:摄像头
    alert_email: str = None  # 新增：报警邮箱
    detection_type: int = 1  # 默认检测类型，现在两个模型都会运行

class Database:
    def __init__(self, db_name: str = "monitor.db"):
        # 获取当前文件（db.py）所在的目录路径（即model目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 拼接得到数据库文件的完整路径（model目录 + 数据库文件名）
        db_path = os.path.join(current_dir, db_name)
        # 连接到正确路径的数据库
        self.conn = sqlite3.connect(db_path)
        self._create_tables()  # 保持原有的表创建逻辑

    def _create_tables(self):
        """创建场景表、视频源表和邮箱表"""
        cursor = self.conn.cursor()

        # 场景表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS scene (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        ''')

        # 邮箱表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS email (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE
        )
        ''')

        # 视频源表：新增detection_type字段（默认检测类型，现在两个模型都会运行）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                isTrue INTEGER NOT NULL DEFAULT 0,  
                scene_id INTEGER NOT NULL,
                type INTEGER NOT NULL,
                is_valid INTEGER NOT NULL DEFAULT 1,
                alert_email TEXT NOT NULL DEFAULT '',
                detection_type INTEGER NOT NULL DEFAULT 1,  -- 默认检测类型，现在两个模型都会运行
                FOREIGN KEY (scene_id) REFERENCES scene (id) ON DELETE CASCADE
            )
            ''')
        
        # 初始化默认邮箱数据
        self._init_default_emails()

        self.conn.commit()
        
        # 检查并添加detection_type字段（如果表已存在但缺少该字段）
        self._add_detection_type_column()
    
    def _add_detection_type_column(self):
        """为已存在的video表添加detection_type字段"""
        cursor = self.conn.cursor()
        try:
            # 检查detection_type字段是否存在
            cursor.execute("PRAGMA table_info(video)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'detection_type' not in columns:
                # 添加detection_type字段
                cursor.execute("ALTER TABLE video ADD COLUMN detection_type INTEGER NOT NULL DEFAULT 1")
                self.conn.commit()
                print("成功添加detection_type字段到video表（现在两个模型都会运行）")
        except Exception as e:
            print(f"添加detection_type字段时出错: {str(e)}")
    
    def _init_default_emails(self):
        """初始化默认邮箱数据，只在表为空时初始化"""
        cursor = self.conn.cursor()
        
        # 检查email表是否为空
        cursor.execute("SELECT COUNT(*) FROM email")
        count = cursor.fetchone()[0]
        
        # 只有当表为空时才初始化默认数据
        if count == 0:
            default_emails = [
                ("管理员1", "903466339@qq.com"),
                # ("管理员2", "Honglingxiang@kaifa.cn"),
                # ("管理员3", "XinHuZhang@kaifa.cn"),
                # ("管理员4", "ShaoHuawang1@kaifa.cn"),
                # ("管理员5", "xiaoyuzhong@kaifa.cn"),
                # ("管理员6", "wqr20011989@163.com")
            ]
            
            for name, email in default_emails:
                try:
                    cursor.execute("INSERT INTO email (name, email) VALUES (?, ?)", (name, email))
                except sqlite3.IntegrityError:
                    pass  # 忽略重复的邮箱
            self.conn.commit()

    # 场景相关操作
    def get_all_scenes(self) -> List[Scene]:
        """获取所有场景"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name FROM scene")
        scenes = [Scene(id=row[0], name=row[1]) for row in cursor.fetchall()]
        return scenes

    def add_scene(self, name: str) -> bool:
        """添加新场景"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO scene (name) VALUES (?)", (name,))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # 场景名称已存在

    def delete_scene(self, scene_id: int) -> bool:
        """删除场景"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM scene WHERE id = ?", (scene_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # 视频源相关操作
    def get_videos_by_scene(self, scene_id: int) -> List[VideoSource]:
        """获取指定场景下的所有视频源"""
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT id, name, path, isTrue, scene_id, type ,is_valid ,alert_email, detection_type
        FROM video 
        WHERE scene_id = ?
        """, (scene_id,))

        videos = [VideoSource(
            id=row[0],
            name=row[1],
            path=row[2],
            is_true=row[3] == 1,
            scene_id=row[4],
            type=row[5],
            is_valid = row[6] == 1 , # 新增：读取有效性标记
            alert_email=row[7],
            detection_type=row[8]
        ) for row in cursor.fetchall()]
        return videos

    def add_video_source(self, video: VideoSource) -> int:
        """添加视频源"""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO video (name, path, isTrue, scene_id, type, is_valid, alert_email, detection_type) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (video.name, video.path, 1 if video.is_true else 0,
              video.scene_id, video.type, 1, video.alert_email, video.detection_type))
        self.conn.commit()
        return cursor.lastrowid

    def update_video_source(self, video: VideoSource) -> bool:
        """更新视频源信息"""
        cursor = self.conn.cursor()
        cursor.execute("""
        UPDATE video 
        SET name = ?, path = ?, isTrue = ?, scene_id = ?, type = ?, is_valid = ?, alert_email = ?, detection_type = ?
        WHERE id = ?
        """, (video.name, video.path, 1 if video.is_true else 0,
              video.scene_id, video.type,  1, video.alert_email, video.detection_type, video.id))
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_video_source(self, video_id: int) -> bool:
        """删除视频源"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM video WHERE id = ?", (video_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def update_video_selection(self, video_id: int, is_true: bool) -> bool:
        """更新视频源的选择状态"""
        cursor = self.conn.cursor()
        cursor.execute("""
        UPDATE video 
        SET isTrue = ? 
        WHERE id = ?
        """, (1 if is_true else 0, video_id))
        self.conn.commit()
        return cursor.rowcount > 0

    """通过ID获取视频源信息"""
    def get_video_by_id(self, video_id: int) -> VideoSource:
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                   SELECT id, name, path, isTrue,is_valid, scene_id, type ,alert_email, detection_type
                   FROM video 
                   WHERE id=?
                   """, (video_id,))
            row = cursor.fetchone()
            if row:
                return VideoSource(
                    id=row[0],
                    name=row[1],
                    path=row[2],
                    is_true=row[3] == 1,
                    is_valid=row[4] == 1,
                    scene_id=row[5],
                    type=row[6],
                    alert_email=row[7],
                    detection_type=row[8]
                )
            return None
        except Exception as e:
            print(f"查询视频源失败: {str(e)}")
            return None

    # 邮箱相关操作
    def get_all_emails(self) -> List[Email]:
        """获取所有可用邮箱"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, email FROM email ORDER BY id")
        emails = [Email(id=row[0], name=row[1], email=row[2]) for row in cursor.fetchall()]
        return emails
    
    def add_email(self, name: str, email: str) -> bool:
        """添加新邮箱"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO email (name, email) VALUES (?, ?)", (name, email))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # 邮箱已存在
    
    def delete_email(self, email_id: int) -> bool:
        """删除邮箱"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM email WHERE id = ?", (email_id,))
        self.conn.commit()
        return cursor.rowcount > 0
    
    def update_email(self, email_id: int, name: str, email: str) -> bool:
        """更新邮箱信息"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE email SET name = ?, email = ? WHERE id = ?", (name, email, email_id))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.IntegrityError:
            return False  # 邮箱已存在
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()
