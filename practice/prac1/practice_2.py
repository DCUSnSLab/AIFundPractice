import cv2
import numpy as np
import tensorflow as tf
import random

# 1. 모델 로드
model = tf.keras.models.load_model("keras_Model.h5", compile=False)

# 2. 클래스 라벨 불러오기
class_names = open("labels.txt", "r", encoding="utf-8").read().splitlines()

# 3. 웹캠 열기
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("영상 창에서 [스페이스] 키를 누르면 현재 프레임을 캡처해 인식합니다.")
print("ESC 키를 누르면 종료합니다.")


if not camera or not camera.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("ESC → 종료 / Space → 가위바위보")

while True:
    ret, frame = camera.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC 종료
        break

    elif key == 32:  # Space → 판정 실행
        # 입력 전처리
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1

        # 예측
        prediction = model.predict(image, verbose=0)
        index = np.argmax(prediction)
        player_choice = class_names[index]
        confidence_score = prediction[0][index]

        # 컴퓨터 랜덤 선택
        computer_choice = random.choice(["rock", "scissors", "paper"])

        # 결과 출력


camera.release()
cv2.destroyAllWindows()
