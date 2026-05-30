# Face Detection Speed Benchmark

Test performed on 10 live webcam frames across 3 separate trial runs to ensure accuracy.

| AI Model | Run 1 Time | Run 2 Time | Run 3 Time | Average Time (10 frames) |
| :--- | :--- | :--- | :--- | :--- |
| **MTCNN** | 0.735s | 0.621s | 0.598s | **0.651 seconds** |
| **RetinaFace (InsightFace)** | 2.001s | 1.843s | 2.014s | **1.953 seconds** |

### Benchmark Conclusion
MTCNN consistently outperforms RetinaFace by approximately 3x on this hardware configuration. MTCNN will be utilized as the primary face detection framework to guarantee a live camera feed operating well above the required 15+ FPS.

## Crop Specification for Recognition Pipeline (cc: Tanish)
* **Format:** RGB
* **Crop Size:** 112x112 pixels