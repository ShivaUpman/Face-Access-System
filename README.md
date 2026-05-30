# Face Access System

A real-time facial recognition access control system. This repository houses the machine learning pipelines for face detection, alignment, and recognition.

## Project Structure

```text
Face-Access-System/
├── ml/
│   └── detection/
│       ├── requirements.txt       # Core dependencies (OpenCV, PyTorch, etc.)
│       ├── capture.py             # Live camera feed test with FPS tracking
│       ├── detect.py              # Live face detection using MTCNN
│       ├── benchmark.py           # Execution speed performance tester
│       └── benchmark_results.md   # Performance metrics (MTCNN vs RetinaFace)
└── README.md                      # Project documentation
```

## 1: Face Detection Pipeline

The detection module utilizes **MTCNN** (Multi-task Cascaded Convolutional Networks) to capture frames from a local webcam, isolate facial boundaries, and identify key facial landmarks in real time.

### System Prerequisites
* Python 3.11 (Official installer version recommended)
* Dedicated Virtual Environment (`venv`) enabled to isolate dependencies

### Installation & Setup

1. **Navigate to the repository root:**
   ` ` `bash
   cd Face-Access-System
   ` ` `

2. **Activate the isolated virtual environment:**
   ` ` `powershell
   # On Windows:
   .\venv\Scripts\Activate.ps1
   
   # On Mac/Linux:
   source venv/bin/activate
   ` ` `

3. **Install the required packages:**
   ` ` `bash
   pip install -r ml/detection/requirements.txt
   ` ` `

### Execution Scripts

* **Test Live Camera Feed (Verify 15+ FPS constraint):**
  ` ` `bash
  python ml/detection/capture.py
  ` ` `

* **Run Real-Time Face & Landmark Tracking:**
  ` ` `bash
  python ml/detection/detect.py
  ` ` `

* **Execute Detection Model Benchmark:**
  ` ` `bash
  python ml/detection/benchmark.py
  ` ` `

*Detailed performance logs and comparison metrics can be reviewed directly in [`ml/detection/benchmark_results.md`](ml/detection/benchmark_results.md).*
