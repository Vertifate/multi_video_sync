import os
import sys
from typing import List, Dict

from PyQt5 import QtCore, QtGui, QtWidgets


class LinkedGraphicsView(QtWidgets.QGraphicsView):
    """
    可缩放、可平移的视图，支持和另一视图联动（同步平移与缩放）。
    通常把左侧视图作为“主视图”，右侧视图作为“从视图”，
    只在主视图的交互事件里去驱动两边的联动。
    """

    def __init__(self, *args, linked_view: "LinkedGraphicsView" = None, is_master: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._linked_view = linked_view
        self._is_master = is_master

        self._panning = False
        self._last_mouse_pos = QtCore.QPoint()

        # 提高平滑显示效果
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        # 使用滚轮缩放、手动平移
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

        # 缩放锚点：使用视图中心，保证双视图缩放时中心一致，避免左右画面产生偏移
        # （如果使用 AnchorUnderMouse，左/右视图的“鼠标位置”不同，就会导致缩放中心不一致，从而出现偏移感）
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)

    def set_linked_view(self, other: "LinkedGraphicsView"):
        self._linked_view = other

    def set_master(self, is_master: bool):
        self._is_master = is_master

    # 缩放逻辑：滚轮放大/缩小
    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if not self._is_master:
            # 从视图不处理交互，全部交给主视图
            super().wheelEvent(event)
            return

        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        # 对当前视图缩放
        self.scale(zoom_factor, zoom_factor)

        # 同步到另一视图
        if self._linked_view is not None:
            self._linked_view.scale(zoom_factor, zoom_factor)

    # 平移逻辑：按住左键拖动
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._is_master and event.button() == QtCore.Qt.LeftButton:
            self._panning = True
            self._last_mouse_pos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._is_master and self._panning:
            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()

            # 平移当前视图
            self._pan_by(delta)

            # 同步平移到另一视图
            if self._linked_view is not None:
                self._linked_view._pan_by(delta)

            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._is_master and event.button() == QtCore.Qt.LeftButton:
            self._panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _pan_by(self, delta: QtCore.QPoint):
        # 通过调整滚动条实现平移
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())


class ImageCompareWindow(QtWidgets.QMainWindow):
    """
    主窗口：
    - 左侧显示 src 图像
    - 右侧显示 frame01 ~ frame06 中选择的一张
    - 左侧支持拖动 & 滚轮缩放，右侧联动
    """

    FRAME_DIRS = ["frame01", "frame02", "frame03", "frame04", "frame05", "frame06"]

    def __init__(self, base_data_dir: str):
        super().__init__()
        self.base_data_dir = base_data_dir
        self.src_dir = os.path.join(base_data_dir, "src")

        self.setWindowTitle("多视图图像细节对比工具")
        self.resize(1400, 800)

        # 数据结构：{filename: {"src": path, "frame01": path, ...}}
        self.image_groups: Dict[str, Dict[str, str]] = {}
        self.current_filename: str = ""
        self.current_frame_key: str = "frame01"

        self._init_ui()
        self._scan_images()
        self._populate_image_list()
        self._update_buttons_state()

        # 默认显示第一张
        if self.image_groups:
            first_name = sorted(self.image_groups.keys())[0]
            self.image_combo.setCurrentText(first_name)
            self.on_image_changed(first_name)

    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)

        # 顶部：图像文件选择 + 复位按钮 + 图表查看按钮
        top_bar = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("图像文件：")
        self.image_combo = QtWidgets.QComboBox()
        self.image_combo.currentTextChanged.connect(self.on_image_changed)

        top_bar.addWidget(label)
        top_bar.addWidget(self.image_combo, 1)
        # 复位视图按钮
        self.reset_view_btn = QtWidgets.QPushButton("复位视图")
        self.reset_view_btn.clicked.connect(self.reset_views_to_fit)
        top_bar.addWidget(self.reset_view_btn)
        # 图表查看按钮
        self.show_chart_btn = QtWidgets.QPushButton("图表查看")
        self.show_chart_btn.clicked.connect(self.show_chart_view)
        top_bar.addWidget(self.show_chart_btn)
        main_layout.addLayout(top_bar)

        # 中间区域使用 QStackedWidget：页面0为图像对比界面，页面1为图表查看界面
        self.stacked = QtWidgets.QStackedWidget()
        main_layout.addWidget(self.stacked, 1)

        # --- 图像对比页面 ---
        image_page = QtWidgets.QWidget()
        image_layout = QtWidgets.QHBoxLayout(image_page)

        # 使用 QSplitter 让左右视图可以调节大小
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        image_layout.addWidget(splitter, 1)

        # 左视图
        self.left_scene = QtWidgets.QGraphicsScene(self)
        self.left_view = LinkedGraphicsView()
        self.left_view.setScene(self.left_scene)

        # 右视图
        self.right_scene = QtWidgets.QGraphicsScene(self)
        self.right_view = LinkedGraphicsView()
        self.right_view.setScene(self.right_scene)

        # 设置联动关系：左为主视图
        self.left_view.set_linked_view(self.right_view)
        self.left_view.set_master(True)
        self.right_view.set_master(False)

        splitter.addWidget(self.left_view)
        splitter.addWidget(self.right_view)
        splitter.setSizes([700, 700])

        # 右侧 frame 按钮（竖直布局）
        right_side = QtWidgets.QVBoxLayout()
        image_layout.addLayout(right_side)

        # 顶部当前帧提示
        self.current_frame_label = QtWidgets.QLabel("当前对比帧：frame01")
        right_side.addWidget(self.current_frame_label)

        frame_label = QtWidgets.QLabel("切换对比帧：")
        right_side.addWidget(frame_label)

        self.frame_buttons: List[QtWidgets.QPushButton] = []
        for idx, frame_dir in enumerate(self.FRAME_DIRS):
            btn = QtWidgets.QPushButton(frame_dir)
            btn.setCheckable(True)
            btn.clicked.connect(self._make_frame_button_handler(frame_dir))
            self.frame_buttons.append(btn)
            right_side.addWidget(btn)

        right_side.addStretch(1)

        # 初始选中 frame01
        if self.frame_buttons:
            self.frame_buttons[0].setChecked(True)
            # 初始化提示文本
            self.current_frame_label.setText(f"当前对比帧：{self.FRAME_DIRS[0]}")

        self.stacked.addWidget(image_page)  # index 0

        # --- 图表查看页面（占位） ---
        chart_page = QtWidgets.QWidget()
        chart_layout = QtWidgets.QVBoxLayout(chart_page)

        chart_label = QtWidgets.QLabel("图表查看界面（待接入真实图表数据）")
        chart_label.setAlignment(QtCore.Qt.AlignCenter)
        chart_layout.addWidget(chart_label, 1)

        back_btn = QtWidgets.QPushButton("返回图片查看")
        back_btn.clicked.connect(self.show_image_view)
        chart_layout.addWidget(back_btn, 0, alignment=QtCore.Qt.AlignCenter)

        self.stacked.addWidget(chart_page)  # index 1

    def _scan_images(self):
        """
        在 src 目录中扫描所有图像文件，并检查在每个 frameXX 中是否存在同名文件。
        只要 src 有，就认为是一个组；frame 缺失则该 frame 不可用。
        """
        if not os.path.isdir(self.src_dir):
            QtWidgets.QMessageBox.warning(self, "错误", f"src 目录不存在：{self.src_dir}")
            return

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        for name in os.listdir(self.src_dir):
            base, ext = os.path.splitext(name)
            if ext.lower() not in valid_exts:
                continue

            group = {"src": os.path.join(self.src_dir, name)}

            # 为每个 frameXX 检查同名文件
            for frame_dir in self.FRAME_DIRS:
                f_dir = os.path.join(self.base_data_dir, frame_dir)
                f_path = os.path.join(f_dir, name)
                if os.path.isfile(f_path):
                    group[frame_dir] = f_path

            self.image_groups[name] = group

    def _populate_image_list(self):
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        for name in sorted(self.image_groups.keys()):
            self.image_combo.addItem(name)
        self.image_combo.blockSignals(False)

    def _update_buttons_state(self):
        """
        根据当前文件是否存在对应 frame 图像，启用/禁用按钮。
        """
        group = self.image_groups.get(self.current_filename, {})
        for btn, frame_dir in zip(self.frame_buttons, self.FRAME_DIRS):
            has_image = frame_dir in group
            btn.setEnabled(has_image)
            if not has_image and btn.isChecked():
                btn.setChecked(False)

    def _make_frame_button_handler(self, frame_key: str):
        def handler():
            if not self.sender().isChecked():
                # 避免取消选中，保证总有一个处于选中状态
                self.sender().setChecked(True)
                return

            # 取消其他按钮的选中状态
            for btn in self.frame_buttons:
                if btn is not self.sender():
                    btn.setChecked(False)

            self.current_frame_key = frame_key
            # 更新右侧提示文本
            self.current_frame_label.setText(f"当前对比帧：{frame_key}")
            self._update_right_image()

        return handler

    # 槽函数：当选择的图像文件改变时
    def on_image_changed(self, filename: str):
        if not filename:
            return
        self.current_filename = filename
        self._update_buttons_state()
        self._update_left_image()
        self._update_right_image(reset_view=True)

    def _load_pixmap(self, path: str) -> QtGui.QPixmap:
        pixmap = QtGui.QPixmap(path)
        if pixmap.isNull():
            QtWidgets.QMessageBox.warning(self, "加载失败", f"无法加载图像：{path}")
        return pixmap

    def _update_left_image(self):
        self.left_scene.clear()
        group = self.image_groups.get(self.current_filename, {})
        src_path = group.get("src")
        if not src_path:
            return
        pixmap = self._load_pixmap(src_path)
        if pixmap.isNull():
            return
        self.left_scene.addPixmap(pixmap)
        # setSceneRect 需要 QRectF，这里将 QRect 转为 QRectF
        self.left_scene.setSceneRect(QtCore.QRectF(pixmap.rect()))

    def _update_right_image(self, reset_view: bool = False):
        self.right_scene.clear()
        group = self.image_groups.get(self.current_filename, {})
        img_path = group.get(self.current_frame_key)
        if not img_path:
            return

        pixmap = self._load_pixmap(img_path)
        if pixmap.isNull():
            return
        self.right_scene.addPixmap(pixmap)
        # setSceneRect 需要 QRectF，这里将 QRect 转为 QRectF
        self.right_scene.setSceneRect(QtCore.QRectF(pixmap.rect()))

        if reset_view:
            # 初次加载或需要复位时，重置两边视图
            self.reset_views_to_fit()

    def reset_views_to_fit(self):
        """
        将左右两个视图复位：取消所有缩放和平移，使整张图以合适比例显示在视图中。
        """
        # 左视图
        self.left_view.resetTransform()
        if not self.left_scene.itemsBoundingRect().isNull():
            self.left_view.fitInView(self.left_scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

        # 右视图
        self.right_view.resetTransform()
        if not self.right_scene.itemsBoundingRect().isNull():
            self.right_view.fitInView(self.right_scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

    # --- 界面切换相关 ---
    def show_chart_view(self):
        """切换到图表查看界面（占位）。"""
        self.stacked.setCurrentIndex(1)

    def show_image_view(self):
        """返回图片查看界面。"""
        self.stacked.setCurrentIndex(0)


def main():
    app = QtWidgets.QApplication(sys.argv)

    # 简单的深色主题美化
    app.setStyle("Fusion")
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 48))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 48))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(63, 63, 70))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0, 122, 204))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
    app.setPalette(dark_palette)

    # 统一按钮、下拉框等控件的圆角与悬停效果
    app.setStyleSheet("""
        QWidget {
            font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            font-size: 11pt;
            color: #FFFFFF;
        }
        QMainWindow {
            background-color: #2D2D30;
        }
        QLabel {
            color: #FFFFFF;
        }
        QComboBox, QLineEdit {
            background-color: #3C3C3C;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 2px 6px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QPushButton {
            background-color: #3E3E42;
            border: 1px solid #555555;
            border-radius: 6px;
            padding: 4px 10px;
        }
        QPushButton:hover {
            background-color: #4E4E52;
        }
        QPushButton:pressed {
            background-color: #007ACC;
        }
        QPushButton:disabled {
            background-color: #3A3A3A;
            color: #777777;
            border-color: #444444;
        }
        QPushButton:checked {
            background-color: #007ACC;
            border-color: #3399FF;
        }
        QSplitter::handle {
            background-color: #444444;
        }
        QGraphicsView {
            background-color: #252526;
            border: 1px solid #3C3C3C;
        }
    """)

    # 假定数据目录结构：
    # base_data_dir/
    #   src/
    #   frame01/
    #   ...
    # 默认以脚本所在目录的 data 目录为根
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, "data")

    base_data_dir = default_data_dir
    if len(sys.argv) > 1:
        # 可通过命令行参数指定 data 目录
        base_data_dir = sys.argv[1]

    window = ImageCompareWindow(base_data_dir)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


