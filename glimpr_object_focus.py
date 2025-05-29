import cv2
import torch
import numpy as np
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Constants
PINHOLE_RADIUS = 100
PINHOLE_CENTER = (320, 240)  # Assuming 640x480 resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# State variables
mode = 1  # Start with Mode 1
last_keypress_time = time.time()
auto_scan_delay = 60
extended_delay = 120
x_offset = 0
scroll_speed = 2
scrolling = False

print("Press '1' for Mode 1 (Full view warped into pinhole)")
print("Press '2' for Mode 2 (Pinhole moveable with WASD and auto scan)")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()

    # Handle keypresses
    if key == ord('1'):
        mode = 1
        print("Switched to Mode 1")
    elif key == ord('2'):
        mode = 2
        print("Switched to Mode 2")
        last_keypress_time = current_time
    elif key in [ord('w'), ord('a'), ord('s'), ord('d')]:
        last_keypress_time = current_time
        scrolling = False
        if key == ord('a'):
            x_offset = max(x_offset - 20, 0)
        elif key == ord('d'):
            x_offset = min(x_offset + 20, FRAME_WIDTH)
    elif key == ord('q'):
        break

    # Mode 1: Warp full view into pinhole
    if mode == 1:
        small_frame = cv2.resize(frame, (PINHOLE_RADIUS * 2, PINHOLE_RADIUS * 2))
        pinhole_view = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        x_start = PINHOLE_CENTER[0] - PINHOLE_RADIUS
        y_start = PINHOLE_CENTER[1] - PINHOLE_RADIUS
        pinhole_view[y_start:y_start + 2 * PINHOLE_RADIUS, x_start:x_start + 2 * PINHOLE_RADIUS] = small_frame
        cv2.circle(pinhole_view, PINHOLE_CENTER, PINHOLE_RADIUS, (255, 255, 255), 2)
        cv2.imshow('Glimpr', pinhole_view)

    # Mode 2: Pinhole view with movement and auto scan
    elif mode == 2:
        # Auto-scroll if enough time passed
        if not scrolling and (current_time - last_keypress_time > auto_scan_delay):
            scrolling = True
            print("Auto scan initiated")

        if scrolling:
            x_offset += scroll_speed
            if x_offset + FRAME_WIDTH > frame.shape[1]:
                x_offset = 0  # Loop scroll

        view_frame = np.roll(frame, -x_offset, axis=1)

        mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        cv2.circle(mask, PINHOLE_CENTER, PINHOLE_RADIUS, 255, -1)
        masked = cv2.bitwise_and(view_frame, view_frame, mask=mask)
        cv2.imshow('Glimpr', masked)

cap.release()
cv2.destroyAllWindows()