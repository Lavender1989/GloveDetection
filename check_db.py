import sqlite3
import os

# 检查数据库文件是否存在
db_path = os.path.join(os.path.dirname(__file__), 'model/monitor.db')

if not os.path.exists(db_path):
    print(f"数据库文件不存在: {db_path}")
    exit(1)

# 连接数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 获取所有表
print("数据库表:")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
for table in tables:
    print(f"  - {table[0]}")

# 检查video表
if 'video' in [t[0] for t in tables]:
    print("\n视频配置:")
    cursor.execute("SELECT * FROM video LIMIT 5")
    videos = cursor.fetchall()
    if videos:
        for video in videos:
            print(f"  - ID: {video[0]}, 名称: {video[1]}, 路径: {video[2]}, 状态: {video[3]}")
    else:
        print("  视频表为空")

# 检查scene表
if 'scene' in [t[0] for t in tables]:
    print("\n场景配置:")
    cursor.execute("SELECT * FROM scene LIMIT 5")
    scenes = cursor.fetchall()
    if scenes:
        for scene in scenes:
            print(f"  - ID: {scene[0]}, 名称: {scene[1]}")
    else:
        print("  场景表为空")

# 关闭连接
conn.close()
