import cv2
import time

def main():
    
    cap = cv2.VideoCapture(0) # default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Variables to calculate FPS
    prev_time = 0

    print("Starting live feed... Press 'Q' to quit.")

    while True:
        # Read the current frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # FPS Calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # FPS number in the top left corner (Green text)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the video window
        cv2.imshow('Live Camera Feed', frame)

        # Wait 1 millisecond for user input, break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up and close everything
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()