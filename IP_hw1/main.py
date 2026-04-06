import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QFileDialog, QWidget, QGridLayout, QVBoxLayout, QGroupBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt

class ImageProcessingHW(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2025 Image Processing HW1")
        self.setGeometry(100, 100, 1000, 800)
        self.img = None  # 儲存目前載入的圖片 (OpenCV format, BGR)

        self.initUI()

    def initUI(self):
        # 主 Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 整體佈局：左邊是按鈕，右邊是顯示區域
        main_layout = QGridLayout()
        central_widget.setLayout(main_layout)

        # 左側按鈕區 
        btn_layout = QVBoxLayout()
        
        self.btn_load = QPushButton("Load Image")
        self.btn_load.setFixedSize(150, 50)
        self.btn_load.clicked.connect(self.load_image)
        
        self.btn_smooth = QPushButton("Smooth Filter")
        self.btn_smooth.setFixedSize(150, 50)
        self.btn_smooth.clicked.connect(self.smooth_filter_task)
        
        self.btn_sharp = QPushButton("Sharp")
        self.btn_sharp.setFixedSize(150, 50)
        self.btn_sharp.clicked.connect(self.sharp_task)
        
        self.btn_gaussian = QPushButton("Gaussian")
        self.btn_gaussian.setFixedSize(150, 50)
        self.btn_gaussian.clicked.connect(self.gaussian_task)
        
        self.btn_lowpass = QPushButton("Lower-pass")
        self.btn_lowpass.setFixedSize(150, 50)
        self.btn_lowpass.clicked.connect(self.lowpass_task)

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_smooth)
        btn_layout.addWidget(self.btn_sharp)
        btn_layout.addWidget(self.btn_gaussian)
        btn_layout.addWidget(self.btn_lowpass)
        btn_layout.addStretch()

        main_layout.addLayout(btn_layout, 0, 0, 2, 1) # 佔據左側

        # 右側圖片顯示區 (2x2 Grid) ---
        # 定義四個顯示區塊
        self.labels = []
        self.titles = []
        
        # 建立 2x2 的顯示位置
        positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
        
        for i, pos in enumerate(positions):
            group = QGroupBox()
            layout = QVBoxLayout()
            
            # 圖片標籤
            lbl_img = QLabel()
            lbl_img.setAlignment(Qt.AlignCenter)
            lbl_img.setMinimumSize(300, 300)
            lbl_img.setStyleSheet("background-color: #f0f0f0; border: 1px solid #999;")
            
            # 文字標題
            lbl_title = QLabel("No use")
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("font-size: 14px; font-weight: bold;")
            
            layout.addWidget(lbl_img)
            layout.addWidget(lbl_title)
            group.setLayout(layout)
            
            main_layout.addWidget(group, *pos)
            
            self.labels.append(lbl_img)
            self.titles.append(lbl_title)

    def display_image(self, img, index, title):
        """Helper: 將 OpenCV 圖片顯示在指定的 label 上"""
        if img is None:
            return
            
        # 轉換顏色 BGR -> RGB
        if len(img.shape) == 3:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 縮放以適應 Label
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.labels[index].width(), 
            self.labels[index].height(), 
            Qt.KeepAspectRatio
        )
        
        self.labels[index].setPixmap(pixmap)
        self.titles[index].setText(title)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', "Image files (*.jpg *.png)")
        if fname:
            self.img = cv2.imread(fname)

            # 先轉灰階以利後續處理 (濾波通常在灰階演示效果較好)
            self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) 
            
            self.display_image(self.img, 0, "Original Image")
            # 清空其他視窗
            for i in range(1, 4):
                self.labels[i].clear()
                self.titles[i].setText("No use")

   
    # --- 演算法實作 ---
    def smooth_filter_task(self):
        """題目 1: Remove noise (Average, Median, Fourier)"""
        if self.img is None: return

        # 1(a) Average Filter
        img_avg = cv2.blur(self.img_gray, (5, 5))
        
        # 1(a) Median Filter 
        img_median = cv2.medianBlur(self.img_gray, 5)

        # 1(b) Fourier Transform Denoise (Ideal Low Pass)
        dft = np.fft.fft2(self.img_gray)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = self.img_gray.shape
        crow, ccol = rows//2, cols//2
        
        # 建立 Mask (Low Pass) 保留中心低頻，過濾外圍高頻雜訊
        mask = np.zeros((rows, cols), np.uint8) # 遮罩初始化
        r = 30 # 半徑
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        # 圓形區域生成
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r 
        mask[mask_area] = 1
        
        # 應用 Mask 並反轉換
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_fourier = np.abs(img_back)
        img_fourier = np.uint8(np.clip(img_fourier, 0, 255))

        # 顯示結果 
        self.display_image(self.img_gray, 0, "Original Image")
        self.display_image(img_avg, 1, "1(a) Average Filter")
        self.display_image(img_median, 2, "1(a) Median Filter")
        self.display_image(img_fourier, 3, "1(b) Fourier Transform")

    def sharp_task(self):
        """題目 2: Sharp (Sobel, Fourier)"""
        if self.img is None: return

        # 2(a) Sobel Mask
        # 計算 x, y 方向的梯度
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        grad_x = cv2.Sobel(self.img_gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(self.img_gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        # 結合梯度 (Sobel Edge)
        sobel_edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        # 銳化 = 原始圖片 + 邊緣 
        img_sobel_sharp = cv2.addWeighted(self.img_gray, 1, sobel_edges, 0.5, 0)

        # 2(b) Fourier High Pass Filter
        f = np.fft.fft2(self.img_gray)
        fshift = np.fft.fftshift(f)
        rows, cols = self.img_gray.shape
        crow, ccol = rows//2, cols//2
        
        # High Pass Mask (中心為 0，其餘為 1)
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
         # 圓形區域生成
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
        mask[mask_area] = 0
        
        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_edges = np.abs(img_back)
        img_edges = np.uint8(np.clip(img_edges, 0, 255))
        # 加上邊緣進行銳化
        img_fourier_sharp = cv2.add(self.img_gray, img_edges)

        # 顯示結果 
        self.display_image(self.img_gray, 0, "Original Image")
        self.display_image(img_sobel_sharp, 1, "2(a) Sobel Sharp") # 題目是 Sobel mask to sharp
        self.display_image(img_fourier_sharp, 2, "2(b) Fourier Sharp")
        self.labels[3].clear(); self.titles[3].setText("No use")

    def gaussian_task(self):
        """題目 3: Design a Gaussian filter of 5*5 mask"""
        if self.img is None: return

        # 使用 OpenCV 產生 5x5 Gaussian Kernel
        # sigmaX=0 根據 kernel size 自動計算
        img_gauss = cv2.GaussianBlur(self.img_gray, (5, 5), 0)
        
        
        self.display_image(self.img_gray, 0, "Original Image")
        self.display_image(img_gauss, 1, "3. Gaussian 5x5 Result")
        self.labels[2].clear(); self.titles[2].setText("No use")
        self.labels[3].clear(); self.titles[3].setText("No use")

    def lowpass_task(self):
        """題目 4: Design a lower-pass filter using Fourier transform based on Gaussian"""
        if self.img is None: return
        
        # 實作 Gaussian Low Pass Filter 
        img_float = np.float32(self.img_gray)
        dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = self.img_gray.shape
        crow, ccol = rows//2 , cols//2
        
        # 建立 Gaussian Mask
        # H(u, v) = exp(-D^2 / (2*D0^2))
        x = cv2.getGaussianKernel(cols, 30)
        y = cv2.getGaussianKernel(rows, 30)
        kernel = y * x.T # 外積產生 2D 高斯
        # 縮放 kernel 到 0~1 之間並調整大小
        mask = kernel / np.max(kernel)
        
        # 建立 2 channel mask (real, imaginary)
        mask_2ch = np.zeros((rows, cols, 2), np.float32)
        mask_2ch[:, :, 0] = mask
        mask_2ch[:, :, 1] = mask
        
        # 應用濾波
        fshift = dft_shift * mask_2ch
        
        # 逆轉換
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        # 正規化回 0-255
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        img_result = np.uint8(img_back)

        self.display_image(self.img_gray, 0, "Original Image")
        self.display_image(img_result, 1, "4. Gaussian Low-pass (Freq)")
        self.labels[2].clear(); self.titles[2].setText("No use")
        self.labels[3].clear(); self.titles[3].setText("No use")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessingHW()
    window.show()
    sys.exit(app.exec_())