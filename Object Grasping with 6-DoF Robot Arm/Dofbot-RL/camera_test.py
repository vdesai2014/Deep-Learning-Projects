import cv2

# Open the camera (assuming it's the first camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Unable to read camera feed")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to 64x64 pixels
    small_frame = cv2.resize(frame, (64, 64))

    # Display the resulting frame
    cv2.imshow('Downsampled Video Feed', small_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
