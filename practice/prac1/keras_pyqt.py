import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        # 모델 로드
        self.model = tf.keras.models.load_model("keras_Model.h5", compile=False)
        self.class_names = open("labels.txt", "r", encoding="utf-8").read().splitlines()

        # 카메라
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # UI 요소
        self.label = QLabel("카메라 화면")
        self.btn = QPushButton("인식하기")
        self.result_label = QLabel("결과: -")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        # 타이머 → 주기적으로 프레임 읽어오기
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 마다 실행

        # 버튼 클릭 → 캡처 후 예측
        self.btn.clicked.connect(self.capture_and_predict)

        self.current_frame = None

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            # OpenCV BGR → RGB 변환
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qt_image))

    def capture_and_predict(self):
        if self.current_frame is None:
            return
        # 모델 입력 전처리
        image = cv2.resize(self.current_frame, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1

        prediction = self.model.predict(image)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]

        self.result_label.setText(f"결과: {class_name} ({confidence_score*100:.2f}%)")

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CameraApp()
    win.setWindowTitle("PyQt5 Camera Inference")
    win.resize(640, 480)
    win.show()
    sys.exit(app.exec_())
