from ultralytics import YOLO
import cv2

model = YOLO("yolo11m.pt")

video_path = "./video.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4",fourcc,fps,(420, 750))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame,persist=True,
        tracker="bytetrack.yaml")

    annotated_frame = results[0].plot()

    resized_frame = cv2.resize(annotated_frame, (420, 750))

    out.write(resized_frame)

    cv2.imshow("YOLO11m Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()         
cv2.destroyAllWindows()