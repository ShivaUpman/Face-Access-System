import cv2
import torch
from facenet_pytorch import MTCNN

def main():
    # This checks if your computer has a dedicated graphics card for AI, otherwise uses the CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Loading AI model on: {device}. Please wait a moment...")

    # Initialize the face detector
    # keep_all=True means it will find multiple faces if there are other people in the background
    mtcnn = MTCNN(keep_all=True, device=device)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Camera active. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # AI models require RGB colors, but OpenCV uses BGR. We have to convert it.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and their landmarks (eyes, nose, mouth corners)
        boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)

        # If the AI found at least one face, draw the boxes and dots
        if boxes is not None:
            for box, landmark in zip(boxes, landmarks):
                # Draw a Green Rectangle around the face
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw the 5 landmark dots
                for point in landmark:
                    x, y = int(point[0]), int(point[1])
                    # Using red dots (0, 0, 255) so they stand out clearly!
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Show the video feed
        cv2.imshow('Live Face Detection', frame)

        # Wait for 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()