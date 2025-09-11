import sys
import traceback
from PyQt6.QtWidgets import QApplication
from controller.main_controller import MainController
from view.main_window import MainWindow

def excepthook(exc_type, exc_value, exc_tb):
    """全局异常处理"""
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(f"全局异常捕获:\n{tb}")
    # 可以在这里记录到文件或显示错误对话框
    QApplication.quit()


if __name__ == "__main__":
    sys.excepthook = excepthook
    app = QApplication(sys.argv)

    try:
        # 先创建窗口
        window = MainWindow()
        # 再创建控制器并传入窗口
        controller = MainController(window)
        # 将控制器设置到窗口
        window.controller = controller
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"主程序异常: {str(e)}")
        traceback.print_exc()