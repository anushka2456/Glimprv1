import cv2
import numpy as np

# Try opening webcam (use CAP_DSHOW for better Windows support)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Error: Webcam not accessible.")
    exit()

# Wait until a valid frame is grabbed
while True:
    ret, frame = cap.read()
    if ret and frame is not None:
        break

# Get frame dimensions
height, width = frame.shape[:2]

# Create pinhole mask
def get_pinhole_mask(shape, radius_fraction=0.3):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    center = (shape[1] // 2, shape[0] // 2)
    radius = int(min(shape[:2]) * radius_fraction)
    cv2.circle(mask, center, radius, 255, -1)
    return mask

mask = get_pinhole_mask(frame.shape)


while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue  # Skip to next frame

    # Blur full frame
    blurred = cv2.GaussianBlur(frame, (25, 25), 0)

    # Apply mask: keep only pinhole area clear
    pinhole_view = np.where(mask[:, :, np.newaxis] == 255, frame, blurred)

    cv2.imshow("Glimpr - Pinhole Vision", pinhole_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
