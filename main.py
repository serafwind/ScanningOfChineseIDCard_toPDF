import sys
import os

# 禁用rembg的后端检查警告
os.environ['REMBG_DISABLE_CHECK'] = '1'

# 打包模式：使用exe内置的模型目录（_MEIPASS/models/u2net/u2net.onnx），实现离线运行背景消除
if getattr(sys, 'frozen', False):
    os.environ.setdefault('REMBG_HOME', sys._MEIPASS)

# 尝试导入rembg（深度学习背景消除）
# 注意：必须在导入PyQt5之前导入rembg/onnxruntime，否则DLL加载顺序冲突会导致段错误
REMBG_AVAILABLE = False
try:
    # 检查onnxruntime是否可用
    import onnxruntime
    print(f"onnxruntime版本: {onnxruntime.__version__}")

    from rembg import remove, new_session
    REMBG_AVAILABLE = True
    print("rembg深度学习库加载成功")
except Exception as e:
    REMBG_AVAILABLE = False
    print(f"警告：rembg不可用，深度学习背景消除功能已禁用")
    print(f"错误详情: {str(e)}")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QMessageBox, QGroupBox, QProgressBar, QCheckBox)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
import cv2
import numpy as np
from PIL import Image
import img2pdf
from io import BytesIO


class ImageLabel(QLabel):
    """可点击选择角点的标签控件"""
    def __init__(self, parent=None, upload_callback=None):
        super().__init__(parent)
        self.points = []
        self.current_image = None
        self.upload_callback = upload_callback
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 200)
        self.setStyleSheet("border: 2px dashed #999; background-color: #f0f0f0;")
        self.setScaledContents(False)

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        self.current_image = pixmap

    def mouseDoubleClickEvent(self, event):
        """双击上传图片"""
        if self.upload_callback:
            self.upload_callback()

    def mousePressEvent(self, event):
        if self.pixmap() and len(self.points) < 4:
            # 计算图片在label中的实际位置
            label_size = self.size()
            pixmap_size = self.pixmap().size()

            # 计算偏移量（居中显示）
            x_offset = (label_size.width() - pixmap_size.width()) // 2
            y_offset = (label_size.height() - pixmap_size.height()) // 2

            # 获取点击位置相对于图片的坐标
            x = event.pos().x() - x_offset
            y = event.pos().y() - y_offset

            # 检查是否在图片范围内
            if 0 <= x < pixmap_size.width() and 0 <= y < pixmap_size.height():
                # 转换到原始图片坐标
                scale_x = self.current_image.width() / pixmap_size.width()
                scale_y = self.current_image.height() / pixmap_size.height()

                original_x = int(x * scale_x)
                original_y = int(y * scale_y)

                self.points.append((original_x, original_y))
                self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap() and self.points:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 5))

            # 计算缩放比例
            label_size = self.size()
            pixmap_size = self.pixmap().size()
            x_offset = (label_size.width() - pixmap_size.width()) // 2
            y_offset = (label_size.height() - pixmap_size.height()) // 2

            scale_x = pixmap_size.width() / self.current_image.width()
            scale_y = pixmap_size.height() / self.current_image.height()

            # 绘制点
            for i, (x, y) in enumerate(self.points):
                screen_x = int(x * scale_x) + x_offset
                screen_y = int(y * scale_y) + y_offset
                painter.drawPoint(screen_x, screen_y)

                # 绘制序号
                painter.drawText(screen_x + 5, screen_y - 5, str(i + 1))

            # 绘制连线
            if len(self.points) > 1:
                for i in range(len(self.points)):
                    p1 = self.points[i]
                    p2 = self.points[(i + 1) % len(self.points)]

                    x1 = int(p1[0] * scale_x) + x_offset
                    y1 = int(p1[1] * scale_y) + y_offset
                    x2 = int(p2[0] * scale_x) + x_offset
                    y2 = int(p2[1] * scale_y) + y_offset

                    painter.drawLine(x1, y1, x2, y2)

    def clear_points(self):
        self.points = []
        self.update()


class IDCardScanner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.front_image = None
        self.back_image = None
        self.front_cv = None
        self.back_cv = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('身份证扫描件生成器')
        self.setGeometry(100, 100, 1200, 800)

        # 主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 标题
        title = QLabel('身份证扫描件生成器')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title)

        # 图片选择区域
        images_layout = QHBoxLayout()

        # 正面图片
        front_group = QGroupBox("身份证正面")
        front_layout = QVBoxLayout(front_group)

        self.front_label = ImageLabel(upload_callback=lambda: self.upload_image('front'))
        self.front_label.setText("双击上传正面图片")
        front_layout.addWidget(self.front_label)

        front_btn_layout = QHBoxLayout()
        self.front_btn = QPushButton('上传正面')
        self.front_btn.clicked.connect(lambda: self.upload_image('front'))
        front_btn_layout.addWidget(self.front_btn)

        self.front_auto_btn = QPushButton('自动校正')
        self.front_auto_btn.clicked.connect(lambda: self.auto_correct('front'))
        self.front_auto_btn.setEnabled(False)
        front_btn_layout.addWidget(self.front_auto_btn)

        self.front_clear_btn = QPushButton('清除角点')
        self.front_clear_btn.clicked.connect(lambda: self.clear_points('front'))
        front_btn_layout.addWidget(self.front_clear_btn)

        front_layout.addLayout(front_btn_layout)
        images_layout.addWidget(front_group)

        # 反面图片
        back_group = QGroupBox("身份证反面")
        back_layout = QVBoxLayout(back_group)

        self.back_label = ImageLabel(upload_callback=lambda: self.upload_image('back'))
        self.back_label.setText("双击上传反面图片")
        back_layout.addWidget(self.back_label)

        back_btn_layout = QHBoxLayout()
        self.back_btn = QPushButton('上传反面')
        self.back_btn.clicked.connect(lambda: self.upload_image('back'))
        back_btn_layout.addWidget(self.back_btn)

        self.back_auto_btn = QPushButton('自动校正')
        self.back_auto_btn.clicked.connect(lambda: self.auto_correct('back'))
        self.back_auto_btn.setEnabled(False)
        back_btn_layout.addWidget(self.back_auto_btn)

        self.back_clear_btn = QPushButton('清除角点')
        self.back_clear_btn.clicked.connect(lambda: self.clear_points('back'))
        back_btn_layout.addWidget(self.back_clear_btn)

        back_layout.addLayout(back_btn_layout)
        images_layout.addWidget(back_group)

        main_layout.addLayout(images_layout)

        # 说明文字
        help_text = QLabel('使用说明：上传图片后，可点击"自动校正"自动检测身份证边缘，或手动点击图片四个角进行透视校正')
        help_text.setStyleSheet("color: #666; font-size: 12px; padding: 10px;")
        help_text.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(help_text)

        # 选项区域
        options_layout = QHBoxLayout()
        options_layout.addStretch()

        self.remove_bg_checkbox = QCheckBox('启用背景消除（实验性）')
        self.remove_bg_checkbox.setChecked(False)
        self.remove_bg_checkbox.setStyleSheet("font-size: 12px; color: #666;")
        options_layout.addWidget(self.remove_bg_checkbox)

        self.print_scale_checkbox = QCheckBox('打印尺寸补偿（打印出来偏小时勾选）')
        self.print_scale_checkbox.setChecked(False)
        self.print_scale_checkbox.setStyleSheet("font-size: 12px; color: #666;")
        self.print_scale_checkbox.setToolTip(
            '打印机按"适合可打印区域"缩小时勾选此项。\n'
            '勾选后卡片在文档中放大1.041倍，打印后恰好为实物尺寸85.45mm。\n'
            '打印时选"实际大小(100%)"则不需要勾选。')
        options_layout.addWidget(self.print_scale_checkbox)

        options_layout.addStretch()
        main_layout.addLayout(options_layout)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)

        # 生成按钮
        generate_layout = QHBoxLayout()
        generate_layout.addStretch()

        self.generate_btn = QPushButton('生成扫描件')
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px 30px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_pdf)
        generate_layout.addWidget(self.generate_btn)

        generate_layout.addStretch()
        main_layout.addLayout(generate_layout)

        # 状态栏
        self.statusBar().showMessage('准备就绪')

    def upload_image(self, side):
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择图片', '',
            'Image Files (*.png *.jpg *.jpeg *.bmp *.gif)'
        )

        if file_path:
            # 读取图片（支持中文路径）
            try:
                # 使用np.fromfile和cv2.imdecode支持中文路径
                cv_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
                if cv_image is None:
                    QMessageBox.warning(self, '错误', '无法读取图片文件')
                    return
            except Exception as e:
                QMessageBox.warning(self, '错误', f'读取图片失败：{str(e)}')
                return

            # 转换为RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # 保存OpenCV格式
            if side == 'front':
                self.front_cv = cv_image
                self.front_image = rgb_image
                label = self.front_label
                self.front_auto_btn.setEnabled(True)
            else:
                self.back_cv = cv_image
                self.back_image = rgb_image
                label = self.back_label
                self.back_auto_btn.setEnabled(True)

            # 显示图片
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # 缩放以适应标签
            scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
            label.current_image = pixmap
            label.clear_points()

            self.statusBar().showMessage(f'已加载{side == "front" and "正面" or "反面"}图片')

    def clear_points(self, side):
        if side == 'front':
            self.front_label.clear_points()
        else:
            self.back_label.clear_points()
        self.statusBar().showMessage('已清除角点')

    def auto_correct(self, side):
        """自动检测身份证边缘"""
        if side == 'front' and self.front_cv is not None:
            cv_image = self.front_cv
            label = self.front_label
        elif side == 'back' and self.back_cv is not None:
            cv_image = self.back_cv
            label = self.back_label
        else:
            return

        try:
            # 转换为灰度图
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 边缘检测
            edged = cv2.Canny(blurred, 50, 200)

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 找到最大的四边形轮廓
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                for contour in contours[:5]:
                    # 轮廓近似
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                    if len(approx) == 4:
                        # 找到了四边形
                        points = [tuple(point[0]) for point in approx]
                        label.points = points
                        label.update()
                        self.statusBar().showMessage('自动检测成功！如需调整可手动点击角点')
                        return

                # 如果没找到四边形，使用最大轮廓的外接矩形
                max_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(max_contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                points = [tuple(point) for point in box]
                label.points = points
                label.update()
                self.statusBar().showMessage('使用外接矩形，可手动调整角点')
            else:
                QMessageBox.warning(self, '提示', '未检测到明显的边缘，请手动点击四个角点')

        except Exception as e:
            QMessageBox.warning(self, '错误', f'自动检测失败：{str(e)}')

    def order_points(self, pts):
        """按左上、右上、右下、左下顺序排列点"""
        rect = np.zeros((4, 2), dtype="float32")

        # 左上角的和最小，右下角的和最大
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # 右上角的差最小，左下角的差最大
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def four_point_transform(self, image, pts):
        """透视变换"""
        rect = self.order_points(np.array(pts, dtype="float32"))
        (tl, tr, br, bl) = rect

        # 计算新图像的宽度
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # 计算新图像的高度
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # 目标点
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # 透视变换矩阵
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def remove_background(self, image):
        """背景消除、透视校正和边缘优化"""
        height, width = image.shape[:2]

        # 第一步：检测身份证边缘并提取四角
        corners = self.detect_id_card_corners(image)

        # 第二步：如果检测到四角，进行透视变换铺平
        if corners is not None and len(corners) == 4:
            # 透视变换铺平身份证
            flattened = self.four_point_transform(image, corners)
        else:
            # 如果检测失败，使用原图
            flattened = image.copy()

        # 第三步：背景消除
        result = self.remove_background_simple(flattened)

        # 第四步：边缘羽化处理
        result = self.smooth_edges(result)

        return result

    def detect_id_card_corners(self, image):
        """检测身份证的四个角点（使用OpenCV）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 100)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours[:5]:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    if area > (image.shape[0] * image.shape[1] * 0.1):
                        points = [tuple(point[0]) for point in approx]
                        return points

            max_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(max_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            points = [tuple(point) for point in box]
            return points

        return None

    def remove_background_simple(self, image):
        """使用深度学习模型（rembg）进行背景消除"""
        if not REMBG_AVAILABLE:
            # 如果rembg不可用，返回原图
            print("rembg不可用，跳过背景消除")
            return image.copy()

        try:
            print("正在使用深度学习模型进行背景消除...")
            # 转换为PIL Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # 使用rembg进行背景消除（基于U²-Net深度学习模型）
            # 会话只创建一次并复用，避免每次重新加载模型
            if not getattr(self, '_rembg_session', None):
                self._rembg_session = new_session('u2net')
            output = remove(pil_image, session=self._rembg_session)

            # 转换回OpenCV格式
            output_array = np.array(output)

            # 如果有alpha通道，处理透明度
            if output_array.shape[2] == 4:
                # 提取RGB和Alpha通道
                rgb = output_array[:, :, :3]
                alpha = output_array[:, :, 3]

                # 创建白色背景
                white_bg = np.ones_like(rgb) * 255

                # 根据alpha通道混合
                alpha = alpha[:, :, np.newaxis] / 255.0
                result = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            else:
                result = output_array

            # 转换回BGR
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            print("深度学习背景消除完成")
            return result_bgr

        except Exception as e:
            print(f"深度学习背景消除失败: {str(e)}")
            # 如果失败，返回原图
            return image.copy()

    def smooth_edges(self, image):
        """平滑边缘，使过渡更自然（只处理边缘区域）"""
        height, width = image.shape[:2]

        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 找到非白色区域（身份证区域）
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        # 轻微扩展边界
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # 找到边界区域（扩展后与原始的差异）
        border_region = dilated - binary

        # 只对边界区域进行轻微模糊
        if np.any(border_region):
            # 创建结果图像
            result = image.copy()

            # 只对边界像素进行轻微平滑
            for y in range(height):
                for x in range(width):
                    if border_region[y, x] > 0:
                        # 检查周围像素
                        if y > 0 and y < height-1 and x > 0 and x < width-1:
                            # 简单的平均模糊
                            result[y, x] = cv2.mean(image[max(0,y-1):min(height,y+2),
                                                         max(0,x-1):min(width,x+2)])[:3]

            return result

        return image

    def apply_rounded_corners(self, image, radius_mm=3.18):
        """给卡片添加标准圆角（身份证/银行卡标准3.18mm），圆角外区域填充白色"""
        height, width = image.shape[:2]
        # 按卡片实际尺寸比例计算圆角半径（卡片宽度对应实测85.45mm）
        radius = max(1, int(round(width * radius_mm / 85.45)))

        # 4倍超采样绘制圆角蒙版，缩小后获得抗锯齿的平滑边缘
        scale = 4
        w4, h4, r4 = width * scale, height * scale, radius * scale
        mask = np.zeros((h4, w4), dtype=np.uint8)
        cv2.rectangle(mask, (r4, 0), (w4 - r4, h4), 255, -1)
        cv2.rectangle(mask, (0, r4), (w4, h4 - r4), 255, -1)
        for cx, cy in [(r4, r4), (w4 - r4, r4), (r4, h4 - r4), (w4 - r4, h4 - r4)]:
            cv2.circle(mask, (cx, cy), r4, 255, -1)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)

        # 圆角外区域填充白色（与A4白底一致）
        alpha = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        result = image.astype(np.float32) * alpha + 255.0 * (1 - alpha)
        return result.astype(np.uint8)

    def generate_pdf(self):
        """生成PDF"""
        # 检查是否上传了图片
        if self.front_cv is None or self.back_cv is None:
            QMessageBox.warning(self, '提示', '请先上传正面和反面图片')
            return

        try:
            self.progress.setVisible(True)
            self.progress.setValue(0)
            self.generate_btn.setEnabled(False)
            QApplication.processEvents()

            # 处理正面
            self.statusBar().showMessage('处理正面图片...')
            QApplication.processEvents()

            if self.front_label.points and len(self.front_label.points) == 4:
                # 用户手动选择了角点，进行透视变换
                front_processed = self.four_point_transform(self.front_cv, self.front_label.points)
            else:
                # 自动检测角点并透视变换
                front_processed = self.remove_background(self.front_cv)

            # 如果用户勾选了背景消除选项
            if self.remove_bg_checkbox.isChecked():
                front_processed = self.remove_background_simple(front_processed)
            self.progress.setValue(30)
            QApplication.processEvents()

            # 处理反面
            self.statusBar().showMessage('处理反面图片...')
            QApplication.processEvents()

            if self.back_label.points and len(self.back_label.points) == 4:
                # 用户手动选择了角点，进行透视变换
                back_processed = self.four_point_transform(self.back_cv, self.back_label.points)
            else:
                # 自动检测角点并透视变换
                back_processed = self.remove_background(self.back_cv)

            # 如果用户勾选了背景消除选项
            if self.remove_bg_checkbox.isChecked():
                back_processed = self.remove_background_simple(back_processed)
            self.progress.setValue(60)
            QApplication.processEvents()

            # 调整尺寸到身份证标准比例 (85.6mm x 54mm)
            self.statusBar().showMessage('生成PDF...')
            QApplication.processEvents()

            # A4尺寸 (像素, 300 DPI)
            a4_width = 2480
            a4_height = 3508

            # 身份证实际尺寸 (像素, 300 DPI)：按游标卡尺实测 85.45mm x 53.95mm
            id_width = 1009   # 85.45mm @300DPI
            id_height = 637   # 53.95mm @300DPI

            # 打印尺寸补偿：打印机按"适合可打印区域"缩小时勾选，
            # 按实测打印误差放大（82.11->85.45, 51.80->53.95），打印后与实物等大
            if self.print_scale_checkbox.isChecked():
                id_width = int(round(id_width * 85.45 / 82.11))
                id_height = int(round(id_height * 53.95 / 51.80))

            # 调整图片尺寸
            front_resized = cv2.resize(front_processed, (id_width, id_height))
            back_resized = cv2.resize(back_processed, (id_width, id_height))

            # 添加标准圆角（与身份证实物一致，3.18mm）
            front_resized = self.apply_rounded_corners(front_resized)
            back_resized = self.apply_rounded_corners(back_resized)

            # 创建A4白色背景
            a4_image = np.ones((a4_height, a4_width, 3), dtype=np.uint8) * 255

            # 计算居中位置
            y_offset = (a4_height - id_height * 2 - 100) // 2
            x_offset = (a4_width - id_width) // 2

            # 放置正面
            a4_image[y_offset:y_offset+id_height, x_offset:x_offset+id_width] = front_resized

            # 放置反面
            y_offset_back = y_offset + id_height + 100
            a4_image[y_offset_back:y_offset_back+id_height, x_offset:x_offset+id_width] = back_resized

            self.progress.setValue(80)
            QApplication.processEvents()

            # 保存为PDF
            save_path, _ = QFileDialog.getSaveFileName(
                self, '保存PDF', '身份证扫描件.pdf',
                'PDF Files (*.pdf)'
            )

            if save_path:
                # 转换为PIL Image
                a4_rgb = cv2.cvtColor(a4_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(a4_rgb)

                # 保存为PDF
                img_bytes = BytesIO()
                # 写入300DPI元数据，确保img2pdf生成精确A4（210x297mm）页面
                pil_image.save(img_bytes, format='PNG', dpi=(300, 300))
                img_bytes.seek(0)

                with open(save_path, 'wb') as f:
                    f.write(img2pdf.convert(img_bytes))

                self.progress.setValue(100)
                self.statusBar().showMessage('PDF生成成功！')

                # 直接打开文件
                os.startfile(save_path)

        except Exception as e:
            QMessageBox.critical(self, '错误', f'生成PDF失败：{str(e)}')
            self.statusBar().showMessage('生成失败')

        finally:
            self.progress.setVisible(False)
            self.generate_btn.setEnabled(True)


def main():
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle('Fusion')

    # 创建主窗口
    window = IDCardScanner()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()