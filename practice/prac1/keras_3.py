import cv2
import numpy as np
import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r", encoding="utf-8").read().splitlines()

# 카메라 열기
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("영상 창에서 [스페이스] 키를 누르면 현재 프레임을 캡처해 인식합니다.")
print("ESC 키를 누르면 종료합니다.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 영상 출력
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC 종료
        break

    elif key == 32:  # Space 키 → 현재 프레임으로 인식
        # 입력 크기로 resize
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        # 모델 입력 형태 맞추기
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1

        # 예측
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        print("Class:", class_name)
        print("Confidence Score:", f"{confidence_score*100:.2f}%")

camera.release()
cv2.destroyAllWindows()
