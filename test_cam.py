import cv2

cap = cv2.VideoCapture(0)  # Try changing to 1 or -1 if needed

if not cap.isOpened():
    print("❌ Webcam not accessible.")
else:
    print("✅ Webcam accessed successfully!")
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Webcam Test", frame)
        cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
