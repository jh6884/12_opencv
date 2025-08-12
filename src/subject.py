from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture('../data/subject.mp4')
model = YOLO('yolo11n.pt')
traffic_classes = { # 참고용으로 남겨둔 딕셔너리
            0: 'person', 2: 'car', 3: 'motorcycle', 
            5: 'bus', 7: 'truck', 9: 'traffic_light'
        }

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (640, 360))
    results = model(img, classes=7, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow('Yolo', annotated_frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()