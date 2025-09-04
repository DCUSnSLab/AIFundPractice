import cv2, numpy as np
import tensorflow as tf
from collections import deque

# 모델 로드 (0~5 손 모양 분류 또는 MNIST 숫자 분류 모델)
model = tf.keras.models.load_model("keras_model.h5")
INPUT_SIZE = (224, 224)   # 모델 입력 크기
LABELS = ['0','1','2','3','4','5','none']

last_result = None

def preprocess(img):
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict(frame):
    x = preprocess(frame)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])

cap = cv2.VideoCapture(0)
pred_q = deque(maxlen=8)   # 예측 스무딩 버퍼
CONF_THR = 0.6

operands = []   # 피연산자 저장
operator = None

def smooth_label(q):
    if not q: return "?"
    vals, cnts = np.unique(q, return_counts=True)
    return vals[np.argmax(cnts)]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    size = min(h, w) // 2
    cx, cy = w // 2, h // 2
    x1, y1 = cx - size // 2, cy - size // 2
    roi = frame[y1:y1+size, x1:x1+size].copy()
    lbl, conf = predict(roi)
    pred_q.append(lbl if conf >= CONF_THR else "?")
    smoothed = smooth_label(list(pred_q))

    # HUD 표시
    cv2.rectangle(frame, (x1, y1), (x1+size, y1+size), (0, 255, 0), 2)
    cv2.putText(frame, f"Pred: {smoothed}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"Operands: {operands}", (10,65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,0), 2)
    cv2.putText(frame, f"Operator: {operator or ''}", (10,100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,200,100), 2)
    if last_result is not None:
        cv2.putText(frame, f"Result: {last_result}",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0,255,255), 3)

    cv2.imshow("Simple Digit Calculator", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # 종료
        break
    if key == ord('r'):  # 초기화
        operands = []
        operator = None
        last_result = None

    # 스페이스바로 숫자 확정
    if key == 32 and smoothed in LABELS:
        if smoothed == 'none':
            continue
        elif len(operands) < 2:
            print(f"예측 값: {smoothed}")

            operands.append(int(smoothed))

    # 연산자 입력 (+, -, *, /)
    if key in (ord('+'), ord('-'), ord('*'), ord('/')) and len(operands) == 2:
        operator = chr(key)
        a, b = operands
        if operator == '+':
            result = a + b
        elif operator == '-':
            result = a - b
        elif operator == '*':
            result = a * b
        elif operator == '/':
            result = a / b if b != 0 else 'inf'
        last_result = f"{a} {operator} {b} = {result}"
        print(last_result)

    # 스페이스바로 초기화
    if key == 32 and operator is not None:
        operands = []
        operator = None
        last_result = None

cap.release()
cv2.destroyAllWindows()