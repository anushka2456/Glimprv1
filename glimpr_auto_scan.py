import cv2
import torch
import numpy as np
import time

# Load YOLOv5 model (pretrained on COCO)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

PINHOLE_RADIUS = 150  # radius of pinhole circle in pixels
PINHOLE_DIAMETER = PINHOLE_RADIUS * 2
PINHOLE_CENTER = None  # to be set dynamically based on frame size

IDLE_TIME_TO_START_SCAN = 60  # seconds before auto scan starts
SCAN_SPEED = 2  # pixels per frame sliding speed

last_interaction_time = time.time()
offset_x = 0
scan_dir = 1

cap = cv2.VideoCapture(0)

def create_pinhole_mask(frame_shape, center, radius):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if PINHOLE_CENTER is None:
        PINHOLE_CENTER = (frame.shape[1] // 2, frame.shape[0] // 2)

    idle_time = time.time() - last_interaction_time

    if idle_time >= IDLE_TIME_TO_START_SCAN:
        # Resize full frame so its height == pinhole diameter
        height = PINHOLE_DIAMETER
        scale_ratio = height / frame.shape[0]
        width = int(frame.shape[1] * scale_ratio)
        warped_frame = cv2.resize(frame, (width, height))

        # Sliding window width = pinhole diameter
        max_offset = width - PINHOLE_DIAMETER
        if max_offset < 0:
            max_offset = 0

        # Update sliding offset_x
        offset_x += SCAN_SPEED * scan_dir
        if offset_x > max_offset:
            offset_x = max_offset
            scan_dir = -1
        elif offset_x < 0:
            offset_x = 0
            scan_dir = 1

        # Crop sliding vertical strip from warped frame
        sliding_strip = warped_frame[:, offset_x:offset_x + PINHOLE_DIAMETER]

        # Create black background frame same size as original
        display_frame = np.zeros_like(frame)

        # Calculate top-left corner of where to place sliding strip in display frame
        top_left_x = PINHOLE_CENTER[0] - PINHOLE_RADIUS
        top_left_y = PINHOLE_CENTER[1] - PINHOLE_RADIUS

        # Place sliding strip inside display frame at pinhole center position
        display_frame[top_left_y:top_left_y + PINHOLE_DIAMETER, top_left_x:top_left_x + PINHOLE_DIAMETER] = sliding_strip

        # Create circular mask at pinhole center
        mask = create_pinhole_mask(display_frame.shape, PINHOLE_CENTER, PINHOLE_RADIUS)

        # Apply mask so only pinhole area is visible
        masked_frame = cv2.bitwise_and(display_frame, display_frame, mask=mask)

    else:
        # Normal pinhole centered on center of frame without sliding strip
        mask = create_pinhole_mask(frame.shape, PINHOLE_CENTER, PINHOLE_RADIUS)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Glimpr - Auto-Scan Mode', masked_frame)

    key = cv2.waitKey(30) & 0xFF
    if key != 255:
        last_interaction_time = time.time()
        # Reset sliding when user interacts
        offset_x = 0
        scan_dir = 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
