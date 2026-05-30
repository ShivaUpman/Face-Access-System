import cv2
import time
import torch
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis

def main():
    print("1. Loading AI Models... (InsightFace may take a minute to download weights on the first run!)")
    
    # Load MTCNN
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    # Load RetinaFace (InsightFace)
    # We restrict it to just 'detection' so it doesn't load unnecessary models
    app = FaceAnalysis(allowed_modules=['detection'], providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("2. Capturing 10 live frames from your webcam...")
    cap = cv2.VideoCapture(0)
    frames_bgr = []
    frames_rgb = []
    
    # Warm up the camera
    for _ in range(5):
        cap.read()
        
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            frames_bgr.append(frame)
            # MTCNN requires RGB colors, InsightFace requires BGR
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames_bgr:
        print("Error: Could not read from webcam.")
        return

    print("3. Running MTCNN Speed Test...")
    start_time = time.time()
    for frame in frames_rgb:
        mtcnn.detect(frame)
    mtcnn_time = time.time() - start_time

    print("4. Running RetinaFace Speed Test...")
    start_time = time.time()
    for frame in frames_bgr:
        app.get(frame)
    retina_time = time.time() - start_time

    print("\n" + "="*40)
    print("        BENCHMARK RESULTS")
    print("="*40)
    print(f"MTCNN Total Time:      {mtcnn_time:.3f} seconds")
    print(f"RetinaFace Total Time: {retina_time:.3f} seconds")
    print("="*40)
    
    print("\n✅ TEST COMPLETE. Write these numbers in your benchmark_results.md file!")

if __name__ == '__main__':
    main()