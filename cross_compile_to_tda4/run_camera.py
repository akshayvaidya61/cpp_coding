import cv2

# Open a connection to the webcam (usually index 0 for the default camera)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_number = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame_number %10 == 0:
        frame_name = f"frame_{frame_number}.jpg"
        # Save the frame as an image file
        cv2.imwrite(frame_name, frame)

    frame_number += 1

cap.release()

