from PyQt5 import QtCore, QtGui, QtWidgets
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
import PIL
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from tensorflow.keras.models import load_model
import io
#import cv2

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
model = load_model('/home/bolgoff/braintumor/my_model.keras', compile=False)

SEGMENT_CLASSES = {
    0 : 'НЕТ ОПУХОЛИ',
    1 : 'НЕКРОТИЧЕСКИЙ/ЯДРО',
    2 : 'ЭДЕМА (ОТЁК)',
    3 : 'РАСПРОСТРАНЕНИЕ'
}

VOLUME_SLICES = 100
VOLUME_START_AT = 22
IMG_SIZE=128

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("Сегментация опухоли мозга")
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        MainWindow.setFixedSize(1920, 1080)
        # MainWindow.showFullScreen()
        MainWindow.setStyleSheet("background-color: #2E3440;")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)

        self.top_buttons_layout = QtWidgets.QVBoxLayout()
        button_style = (
            "QPushButton {"
            "    color: #FFFFFF;"
            "    background-color: #4C566A;"
            "    border-radius: 8px;"
            "    padding: 10px;"
            "    font-size: 22px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #5E81AC;"
            "}"
        )

        self.load_image = QtWidgets.QPushButton("Загрузить изображение")
        self.load_image.setStyleSheet(button_style)
        self.segmentation = QtWidgets.QPushButton("Запустить сегментацию")
        self.segmentation.setStyleSheet(button_style)
        self.save_image = QtWidgets.QPushButton("Сохранить сегментацию")
        self.save_image.setStyleSheet(button_style)
        self.save_report = QtWidgets.QPushButton("Сохранить отчёт")
        self.save_report.setStyleSheet(button_style)

        self.load_image.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.segmentation.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.save_image.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.save_report.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        
        self.top_buttons_layout.addWidget(self.load_image, 1)
        self.top_buttons_layout.addWidget(self.segmentation, 1)
        self.top_buttons_layout.addWidget(self.save_image, 1)
        self.top_buttons_layout.addWidget(self.save_report, 1)

        self.top_buttons_layout.addStretch()
        self.top_buttons_layout.setAlignment(QtCore.Qt.AlignTop)

        self.content_layout = QtWidgets.QHBoxLayout()

        self.content_layout.addLayout(self.top_buttons_layout)

        self.images_widget = QtWidgets.QWidget()
        self.images_widget.setStyleSheet("background-color: #3B4252; border-radius: 10px;")
        self.images_layout = QtWidgets.QGridLayout(self.images_widget)

        self.x_axis = self.create_axis_widget("Осевой срез")
        self.y_axis = self.create_axis_widget("Коронарный срез")
        self.z_axis = self.create_axis_widget("Сагиттальный срез")

        self.images_layout.addWidget(self.x_axis, 0, 0)
        self.images_layout.addWidget(self.y_axis, 0, 1)
        self.images_layout.addWidget(self.z_axis, 1, 0)

        self.three_d_view = gl.GLViewWidget()
        self.three_d_view.setGeometry(0, 0, int(1080/2), int(1920/2))
        self.three_d_view.setCameraPosition(distance=120, elevation=0, azimuth=0)
        self.three_d_view.pan(0, 0, 10)
        
        self.images_layout.addWidget(self.three_d_view, 1, 1)

        self.content_layout.addWidget(self.images_widget)

        self.main_layout.addLayout(self.content_layout)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuFile.menuAction())

        self.load_image.clicked.connect(self.load_mri_image)
        self.save_image.clicked.connect(self.save_segmentation)
        self.save_report.clicked.connect(self.save_report_func)

        self.segmentation.clicked.connect(self.run_segmentation)

    def create_axis_widget(self, label_text):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        label = QtWidgets.QLabel(label_text)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("color: #D8DEE9; font-size: 16px;")
        layout.addWidget(label)

        canvas = FigureCanvas(plt.figure())
        layout.addWidget(canvas)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setStyleSheet(
            "QSlider::groove:horizontal { background: #4C566A; height: 8px; }"
            "QSlider::handle:horizontal { background: #88C0D0; width: 16px; border-radius: 8px; margin: -4px 0; }"
        )
        slider.valueChanged.connect(self.slider_moved)
        layout.addWidget(slider)

        widget.slider = slider
        widget.canvas = canvas
        widget.slice_data = None

        return widget

    def load_mri_image(self):
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            None, "Выберите файл изображения", "", "NIfTI Files (*.nii *.nii.gz)")

        if file_path:
            global mri_data
            mri_data = nib.load(file_path).get_fdata()
            self.display_mri_slices(mri_data)
            self.display_3d_view(np.squeeze(mri_data))

    def display_mri_slices(self, mri_data):
        self.mri_data = mri_data

        axial_middle = mri_data.shape[2] // 2
        coronal_middle = mri_data.shape[1] // 2
        sagittal_middle = mri_data.shape[0] // 2

        self.x_axis.slice_data = mri_data
        self.y_axis.slice_data = mri_data
        self.z_axis.slice_data = mri_data

        self.x_axis.slider.setMaximum(mri_data.shape[2] - 1)
        self.y_axis.slider.setMaximum(mri_data.shape[1] - 1)
        self.z_axis.slider.setMaximum(mri_data.shape[0] - 1)

        self.update_canvas(self.x_axis.canvas, mri_data[:, :, axial_middle])
        self.update_canvas(self.y_axis.canvas, mri_data[:, coronal_middle, :])
        self.update_canvas(self.z_axis.canvas, mri_data[sagittal_middle, :, :])

        self.x_axis.slider.setValue(axial_middle)
        self.y_axis.slider.setValue(coronal_middle)
        self.z_axis.slider.setValue(sagittal_middle)

    def slider_moved(self):
        sender = self.centralwidget.sender()
        if sender == self.x_axis.slider:
            slice_idx = sender.value()
            self.update_canvas(self.x_axis.canvas, self.x_axis.slice_data[:, :, slice_idx])
        elif sender == self.y_axis.slider:
            slice_idx = sender.value()
            self.update_canvas(self.y_axis.canvas, self.y_axis.slice_data[:, slice_idx, :])
        elif sender == self.z_axis.slider:
            slice_idx = sender.value()
            self.update_canvas(self.z_axis.canvas, self.z_axis.slice_data[slice_idx, :, :])

    def update_canvas(self, canvas, image_slice):
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        ax.imshow(image_slice.T, origin="lower", cmap='gray')
        canvas.draw()

    def display_3d_view(self, mri_data):
        RENDER_TYPE = "translucent"
        THR_MIN = 1
        THR_MAX = 2000

        self.three_d_view.clear()

        data = mri_data
        data[data == 0] = THR_MIN
        data[data < THR_MIN] = THR_MIN
        data[data >= THR_MAX] = THR_MAX
        data -= THR_MIN
        data /= THR_MAX - THR_MIN

        data = data[:, ::-1, :]
        half_x = data.shape[0] // 2
        cropped_data = data[:half_x, :, :]
        d2 = np.zeros(cropped_data.shape + (4,))

        d2[..., 3] = cropped_data**1 * 255  # прозрачность
        d2[..., 0] = d2[..., 3]  # красный
        d2[..., 1] = d2[..., 3]  # зеленый
        d2[..., 2] = d2[..., 3]  # синий
        
        d2[:40, 0, 0] = [255, 0, 0, 255]
        d2[0, :40, 0] = [0, 255, 0, 255]
        d2[0, 0, :40] = [0, 0, 255, 255]
        d2 = d2.astype(np.ubyte)

        volume = gl.GLVolumeItem(d2, sliceDensity=6, smooth=False, glOptions=RENDER_TYPE)
        volume.translate(dx=-d2.shape[0]/2, dy=-d2.shape[1]/2, dz=-d2.shape[2]/3)
        self.three_d_view.addItem(volume)

    def save_segmentation(self):
        if not hasattr(self, "mri_data") or self.mri_data is None:
            QtWidgets.QMessageBox.warning(None, "Ошибка", "Нет данных для сохранения!")
            return

        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            None, "Выберите путь для сохранения", "", "NIfTI Files (*.nii *.nii.gz)")

        if file_path:
            nifti_image = nib.Nifti1Image(self.mri_data, affine=np.eye(4))
            try:
                nib.save(nifti_image, file_path)
                QtWidgets.QMessageBox.information(None, "Удачно", "Файл успешно сохранён!")
            except Exception as e:
                QtWidgets.QMessageBox.critical(None, "Ошибка", f"Ошибка при сохранении: {e}")

    def generate_pdf_report(self, file_path):
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        title = Paragraph("Report for Brain Tumor Segmentation", styles["Title"])
        elements.append(title)

        slice_count = f"Layers count: {self.mri_data.shape[2]}"
        elements.append(Paragraph(slice_count, styles["Normal"]))

        middle_slices = [
            self.mri_data[:, :, self.mri_data.shape[2] // 2],
            self.mri_data[:, :, self.mri_data.shape[2] // 4],
            self.mri_data[:, :, 3 * self.mri_data.shape[2] // 4],
        ]

        for i, slice_data in enumerate(middle_slices):
            buf = io.BytesIO()
            plt.figure()
            plt.imshow(slice_data.T, cmap="gray", origin="lower")
            plt.title(f"Layer {i + 1}")
            plt.axis("off")
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)

            img = Image(buf)
            img._restrictSize(400, 300)
            elements.append(img)

        description = """
        This report includes basic MRI image data.,
        as well as visualization of several sections, among which tumors or other pathologies may be visible.
        """

        elements.append(Paragraph(description, styles["Normal"]))

        doc.build(elements)

    def save_report_func(self):
        if not hasattr(self, "mri_data") or self.mri_data is None:
            QtWidgets.QMessageBox.warning(None, "Ошибка", "Нет данных для отчета!")
            return

        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            None, "Выберите путь для сохранения отчета", "", "PDF Files (*.pdf)"
        )

        if file_path:
            try:
                self.generate_pdf_report(file_path)
                QtWidgets.QMessageBox.information(None, "Удачно", "Отчет успешно сохранён!")
            except Exception as e:
                QtWidgets.QMessageBox.critical(None, "Ошибка", f"Ошибка при сохранении: {e}")

    def run_segmentation(self):
        dim = (IMG_SIZE, IMG_SIZE)
        if not hasattr(self, "mri_data") or self.mri_data is None:
            QtWidgets.QMessageBox.warning(None, "Ошибка", "Нет данных для сегментации!")
            return
        
        progress_dialog = QtWidgets.QProgressDialog("Сегментация в процессе...", "Отмена", 0, self.mri_data.shape[2])
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)

        try:
            # Подготовка входных данных
            X = np.zeros((self.mri_data.shape[2], IMG_SIZE, IMG_SIZE, 1))  # Изменил на 1 канал, если у вас одноканальные изображения
            for j in range(self.mri_data.shape[2]):
                img = self.mri_data[:, :, j]
                # Нормализация к диапазону [0, 1] для BRATS
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
                resized_img = PIL.Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE))
                X[j, :, :, 0] = np.array(resized_img)
                progress_dialog.setValue(j + 1)
                if progress_dialog.wasCanceled():
                    break
            progress_dialog.close()
            
            # Предсказание (убрал [0], если модель возвращает массив правильной формы)
            segmented_data = model.predict(X, verbose=1)
            
            # Если segmented_data имеет размерность (N, H, W, C), берем первый канал или argmax
            if segmented_data.ndim == 4:
                segmented_data = np.argmax(segmented_data, axis=-1)  # для многоклассовой сегментации
                # или segmented_data = segmented_data[..., 0]  # если нужен первый канал
                
            self.display_mri_slices(segmented_data)
            self.display_3d_view(segmented_data)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Ошибка", f"Ошибка при сегментации: {e}")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())