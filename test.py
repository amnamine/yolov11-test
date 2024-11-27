from ultralytics import YOLO
import cv2

# Load the YOLO model (replace 'yolo11n.pt' with the correct path to your model file)
model = YOLO('yolo11n.pt')

# Open webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop through frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO detection
    results = model(frame, stream=False)  # Run inference on the current frame

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
