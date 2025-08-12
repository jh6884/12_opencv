from ultralytics import YOLO # type: ignore
import cv2

model = YOLO('yolo11n.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, verbose = False)
    annotated_frame = results[0].plot()
    cv2.imshow('Yolo', annotated_frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()