import cv2
import numpy as np
import tensorflow as tf

# 1. 모델 로드
model = tf.keras.models.load_model("keras_Model.h5", compile=False)

# 2. 클래스 라벨 불러오기
class_names = open("labels.txt", "r", encoding="utf-8").read().splitlines()

# 3. 이미지 파일 불러오기
image_path = "image_dog.jpg"   # 추론할 이미지 경로
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

# 4. 전처리 (모델 입력 크기 맞추기)
image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
image_data = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
image_data = (image_data / 127.5) - 1   # Normalize [-1, 1]

# 5. 추론
prediction = model.predict(image_data)
predictions = prediction[0]

# 6. 전체 클래스별 결과 출력
print("=== 전체 클래스별 확률 ===")
for i, score in enumerate(predictions):
    print(f"{class_names[i]} : {score*100:.2f}%")

# 7. 가장 높은 확률의 클래스 출력
index = np.argmax(predictions)
class_name = class_names[index]
confidence_score = predictions[index]

print("\n=== 최종 결과 ===")
print(f"Class: {class_name}")
print(f"Confidence Score: {confidence_score*100:.2f}%")