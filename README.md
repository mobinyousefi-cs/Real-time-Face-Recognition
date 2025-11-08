# 🎥 Real-time Face Recognition with Python & OpenCV

**Author:** [Mobin Yousefi](https://github.com/mobinyousefi-cs)  
**License:** MIT  
**Version:** 0.1.0

---

## 🧠 Overview

This project demonstrates **real-time human face detection and recognition** using **Python** and **OpenCV**.  
It leverages **Haar Cascade Classifiers** for face detection and the **LBPH (Local Binary Pattern Histogram)** algorithm for face recognition.

Face recognition is a vital computer vision task used in security systems, authentication, and smart monitoring applications.

This project is designed for **beginners to intermediate learners** who want to understand how face recognition works in real time using a webcam.

---

## 🔍 What You’ll Learn

- The basics of **Haar Cascade Classifiers** and **LBPH Face Recognizers**.  
- How to **capture and store facial images** for training.  
- How to **train an LBPH model** to recognize faces.  
- How to **run a real-time recognition loop** on webcam video feed.  

---

## 📂 Project Structure

```
realtime-face-recognition-opencv/
├── src/
│   └── realtime_face_recognition/
│       ├── __init__.py
│       ├── config.py
│       ├── detection.py
│       ├── capture_faces.py
│       ├── train_model.py
│       ├── recognize.py
│       └── cli.py
├── data/
│   ├── raw_faces/
│   └── processed_faces/
├── models/
│   ├── lbph_face_recognizer.xml
│   └── labels.json
├── tests/
│   └── test_imports.py
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mobinyousefi-cs/realtime-face-recognition-opencv.git
cd realtime-face-recognition-opencv
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

This will install the required packages:
- `opencv-contrib-python`
- `numpy`

---

## 🚀 Usage

### 1. Capture Faces for Training

Capture face images for a given person label (e.g., “mobin”).  
Press **`c`** to capture a face and **`q`** to quit.

```bash
realtime-face-recognition capture --label mobin
```

Captured images are saved under:
```
data/raw_faces/mobin/
```

---

### 2. Train the Model

After collecting enough images, train the LBPH recognizer:

```bash
realtime-face-recognition train
```

This creates two files under the `models/` directory:
```
models/lbph_face_recognizer.xml
models/labels.json
```

---

### 3. Run Real-time Face Recognition

Use your webcam to recognize faces live:

```bash
realtime-face-recognition recognize
```

- Green boxes indicate recognized faces.  
- Red boxes indicate unknown faces.  
- Press **`q`** to quit.

You can adjust sensitivity with:

```bash
realtime-face-recognition recognize --confidence-threshold 80
```

---

## 🧩 How It Works

### Haar Cascade Detection
Haar features are rectangular patterns (like convolution filters) used to detect facial regions.  
OpenCV provides pre-trained models for these features.

### LBPH Recognition
The **Local Binary Pattern Histogram** method converts face regions into texture patterns.  
LBPH is simple, efficient, and works well under varying lighting conditions.

---

## 🧠 Key Components

| Module | Description |
|--------|-------------|
| `config.py` | Defines data/model paths and parameters |
| `detection.py` | Handles face detection with Haar cascade |
| `capture_faces.py` | Captures training images from webcam |
| `train_model.py` | Trains and saves the LBPH model |
| `recognize.py` | Real-time recognition loop |
| `cli.py` | Command-line interface entrypoint |

---

## 🧪 Testing

Run quick smoke tests to ensure imports are valid:

```bash
pytest tests/test_imports.py
```

---

## 💡 Tips

- Capture **20–30 images per person** from different angles for best accuracy.  
- Use good lighting and avoid shadows on the face.  
- Keep faces centered and within detection boundaries.  

---

## 🛠 Troubleshooting

| Problem | Possible Fix |
|----------|---------------|
| Webcam not detected | Try changing `--camera-index` to 1 or 2 |
| No faces detected | Ensure lighting is adequate and camera is focused |
| Model not found | Run the training step again after capturing faces |
| Unknown faces | Increase `--confidence-threshold` or add more training samples |

---

## 📘 References

- [OpenCV Haar Cascade Documentation](https://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html)  
- [OpenCV Face Recognition API](https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html)  
- [Local Binary Patterns (LBP) Theory](https://www.pyimagesearch.com/2021/01/19/local-binary-patterns-with-python-opencv/)

---

## 🧑‍💻 Author

**Mobin Yousefi**  
Master’s in Computer Science  
[GitHub: mobinyousefi-cs](https://github.com/mobinyousefi-cs)

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this project helpful, consider starring the repo!**
