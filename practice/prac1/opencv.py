import cv2

cap = cv2.VideoCapture(0)  # 0번 웹캠 사용
while True:
    ret, frame = cap.read()
    cv2.imshow("Webcam Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
