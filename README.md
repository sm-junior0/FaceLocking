# FaceLocking - Real-Time Face Recognition & Tracking System

A CPU-efficient face recognition and tracking system with face locking capabilities. Built using Haar Cascade detection, MediaPipe FaceMesh landmarks, and ArcFace embeddings.

## Features

- **Real-time Face Detection**: Efficient Haar Cascade + MediaPipe FaceMesh pipeline
- **5-Point Facial Landmarks**: Stable landmark extraction (eyes, nose, mouth)
- **Face Alignment**: ArcFace-style alignment to 112x112 normalized crops
- **Face Embedding**: ONNX-based ArcFace embedder for face recognition
- **Face Enrollment**: Interactive enrollment system with auto-capture mode
- **Multi-Face Recognition**: Recognize multiple faces simultaneously
- **Face Locking**: Lock onto and track a specific person
- **Action Detection**: Tracks head movements, eye blinks, and smiles when locked
- **Threshold Evaluation**: Tools to optimize recognition thresholds

## System Requirements

- Python 3.7+
- Webcam (camera index 0 by default)
- CPU (GPU not required - optimized for CPU execution)

## Installation

1. **Clone the repository**
```bash
cd FaceLocking
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download required models**

   Download the following model files and place them in the `models/` directory:
   
   - `embedder_arcface.onnx` - ArcFace face embedding model
   - `face_landmarker.task` - MediaPipe FaceMesh model
   
   > **Note**: The MediaPipe model can be downloaded from [MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)

4. **Initialize project structure**
```bash
python init_project.py
```

This creates the necessary directories:
- `data/enroll/` - Stores enrollment images
- `data/db/` - Stores face database
- `models/` - Contains ML models
- `logs/` - Stores face lock action histories

## Usage

### 1. Camera Test
Test your camera connection:
```bash
python -m src.camera
```
Press `q` to quit.

### 2. Face Detection Demo
Test Haar Cascade face detection:
```bash
python -m src.detect
```
Press `q` to quit.

### 3. Landmark Detection Demo
Test 5-point facial landmark detection:
```bash
python -m src.landmarks
```
Press `q` to quit.

### 4. Face Alignment Demo
Test face alignment to 112x112:
```bash
python -m src.align
```
- Press `q` to quit
- Press `s` to save aligned face snapshot

### 5. Face Embedding Demo
Visualize face embeddings in real-time:
```bash
python -m src.embed
```
- Press `q` to quit
- Press `p` to print embedding statistics

### 6. Face Enrollment
Enroll new faces into the database:
```bash
python -m src.enroll
```

**Controls**:
- `SPACE` - Capture one sample
- `a` - Toggle auto-capture mode (captures every 0.25s)
- `s` - Save enrollment (requires 15+ samples total)
- `r` - Reset NEW samples (keeps existing on disk)
- `q` - Quit

**Tips for good enrollment**:
- Ensure stable lighting
- Move slightly left/right
- Use different facial expressions
- Capture at least 15 samples per person

**Re-enrollment**: If enrollment crops already exist for a person, they will be loaded and included in the template. New captures are appended to existing samples.

### 7. Face Recognition with Locking
Run the main recognition system:
```bash
python -m src.recognize
```

**Controls**:
- `q` - Quit
- `r` - Reload face database from disk
- `+/-` - Adjust recognition threshold
- `d` - Toggle debug overlay
- `l` - Lock/unlock onto recognized face

**Face Locking Features**:
When a face is locked:
- Tracks the person continuously
- Detects and logs head movements (left/right)
- Detects eye blinks
- Detects smiles
- Saves action history to `logs/` when unlocked

**Timeout**: If a locked face is not seen for 2 seconds, the system automatically unlocks and saves the history.

### 8. Threshold Evaluation
Evaluate and optimize recognition thresholds:
```bash
python -m src.evaluate
```

This analyzes enrollment crops to:
- Calculate genuine vs impostor distance distributions
- Suggest optimal threshold for target FAR (False Acceptance Rate)
- Show FAR/FRR (False Rejection Rate) trade-offs

**Requirements**: At least 5 aligned crops per person in `data/enroll/`.

## Project Structure

```
FaceLocking/
│
├── src/
│   ├── camera.py          # Camera test utility
│   ├── detect.py          # Face detection demo (Haar Cascade)
│   ├── landmarks.py       # 5-point landmark detection demo
│   ├── haar_5pt.py        # Haar + FaceMesh detector class
│   ├── align.py           # Face alignment demo
│   ├── embed.py           # Face embedding with ArcFace
│   ├── enroll.py          # Face enrollment tool
│   ├── recognize.py       # Main recognition + face locking
│   └── evaluate.py        # Threshold evaluation tool
│
├── data/
│   ├── enroll/            # Enrollment images (per person subdirs)
│   ├── db/                # Face database (NPZ + JSON)
│   └── debug_aligned/     # Debug: saved aligned faces
│
├── models/
│   ├── embedder_arcface.onnx      # ArcFace embedding model
│   └── face_landmarker.task       # MediaPipe FaceMesh model
│
├── logs/                  # Face lock action histories
├── requirements.txt       # Python dependencies
├── init_project.py        # Project initialization script
└── README.md             # This file
```

## Pipeline Architecture

The face recognition pipeline consists of these stages:

```
Camera Frame
    ↓
[1] Haar Cascade Detection (Fast, rough face localization)
    ↓
[2] MediaPipe FaceMesh (Precise 5-point landmarks)
    ↓
[3] Face Alignment (5-point similarity transform → 112x112)
    ↓
[4] ArcFace Embedding (ONNX → L2-normalized vector)
    ↓
[5] Database Matching (Cosine distance)
    ↓
Recognition Result
```

### Why This Pipeline?

- **Haar Cascade**: Fast, CPU-friendly initial detection
- **MediaPipe FaceMesh**: Confirms real faces, provides stable landmarks
- **5-Point Alignment**: Standard for ArcFace models (eyes, nose, mouth corners)
- **ONNX Runtime**: Cross-platform, efficient CPU inference
- **Cosine Distance**: Natural metric for L2-normalized embeddings

## Face Database Format

The enrollment process creates two files in `data/db/`:

1. **face_db.npz** - NumPy archive containing:
   - One L2-normalized embedding vector per enrolled person
   - Key: person name, Value: 512-dim float32 array

2. **face_db.json** - Metadata including:
   - List of enrolled names
   - Timestamp of last update
   - Embedding dimensionality
   - Sample counts used for each enrollment

## Recognition Metrics

The system uses **cosine distance** for matching:

```
cosine_similarity = dot(embedding_a, embedding_b)  # Since embeddings are L2-normalized
cosine_distance = 1 - cosine_similarity
```

**Default threshold**: `0.34` (distance)
- Lower = stricter matching
- Higher = more permissive matching

Use `evaluate.py` to find optimal thresholds for your use case.

## Action Detection

When a face is locked (`l` key), the system tracks:

1. **Head Movement**:
   - Left/Right detection based on face center displacement
   - Threshold: 10px movement

2. **Eye Blink**:
   - Detects vertical distance changes between eyes and nose
   - Threshold: 30% reduction indicates blink

3. **Smile**:
   - Mouth width/height ratio analysis
   - Threshold: 1.5x increase in ratio

All actions are logged with timestamps to `logs/<person>_history_<timestamp>.txt`.

## Troubleshooting

### Camera not opening
```
RuntimeError: Camera not opened
```
**Solution**: Try changing camera index (0, 1, or 2) in the code or check camera permissions.

### Model not found
```
RuntimeError: Model not found: models/face_landmarker.task
```
**Solution**: Download MediaPipe FaceMesh model and place in `models/` directory.

### No faces detected
**Solutions**:
- Ensure good lighting
- Face the camera directly
- Adjust `min_size` parameter in detector (default: 70x70)
- Check that face is not too close or too far from camera

### Poor recognition accuracy
**Solutions**:
- Enroll more samples per person (20-30 recommended)
- Vary expressions and head angles during enrollment
- Ensure consistent lighting between enrollment and recognition
- Run `evaluate.py` to optimize threshold
- Adjust threshold using `+/-` keys during recognition

### FaceMesh not detecting landmarks
**Solutions**:
- Ensure face is within Haar detection box
- Improve lighting conditions
- Face camera more directly
- Check that `face_landmarker.task` model is correctly installed

## Performance

On a typical CPU (Intel i5/i7):
- **Detection + Landmarks**: ~30-50 FPS
- **Full Pipeline (including embedding)**: ~15-25 FPS
- **Recognition (single face)**: ~20-30 FPS
- **Recognition (multi-face)**: Scales with number of faces

## Technical Details

### Face Alignment
Uses ArcFace-standard 5-point template:
- Input: 5 keypoints (left eye, right eye, nose, mouth corners)
- Output: 112x112 aligned face crop
- Transform: Similarity transform (rotation + scale + translation)

### Embedding Model
- **Input**: 112x112 RGB image
- **Preprocessing**: (pixel - 127.5) / 128.0
- **Output**: 512-dim L2-normalized vector
- **Framework**: ONNX Runtime (CPU provider)

### Smoothing
Temporal smoothing (EMA) applied to:
- Bounding box coordinates (alpha=0.80)
- Keypoint positions (alpha=0.80)

This reduces jitter while maintaining responsiveness.

## Credits & References

- **Haar Cascade**: OpenCV built-in frontal face detector
- **MediaPipe**: Google MediaPipe FaceMesh
- **ArcFace**: Additive Angular Margin Loss for Deep Face Recognition
- **ONNX Runtime**: Cross-platform ML inference engine

## License

This project is intended for educational and research purposes.

## Future Improvements

- [ ] Add GPU support for faster inference
- [ ] Multi-threaded processing for better FPS
- [ ] Web interface for remote enrollment
- [ ] Export/import face database
- [ ] Support for additional action detections
- [ ] Integration with access control systems
- [ ] Face spoofing detection (liveness check)
- [ ] Age/gender estimation
- [ ] Emotion recognition

## Contributing

Contributions are welcome! Areas for improvement:
- Performance optimization
- Additional action detection algorithms
- Better UI/UX for enrollment and recognition
- Documentation and tutorials
- Testing and validation tools

---

**Note**: This system is designed for educational purposes. For production deployment in security-critical applications, additional measures like liveness detection, anti-spoofing, and security audits are recommended.
