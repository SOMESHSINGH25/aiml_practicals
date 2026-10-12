# AI/ML Practicals Using Python

A collection of 10 Artificial Intelligence and Machine Learning practicals implemented using Python. This repository covers fundamental machine learning concepts, including correlation analysis, clustering, dimensionality reduction, classification, regression, regularization, object detection, and multimodal fake news detection.

## 📌 List of Practicals

| No. | Practical | Description |
|---|---|---|
| 1 | Correlation Analysis | Identify the association between dependent and independent variables. |
| 2 | Clustering | Apply unsupervised learning techniques to group similar data objects. |
| 3 | Principal Component Analysis (PCA) | Perform dimensionality reduction using PCA. |
| 4 | Naïve Bayes Classification | Implement a Naïve Bayes classification algorithm. |
| 5 | K-Nearest Neighbors (KNN) | Perform classification using the KNN algorithm. |
| 6 | Linear Regression | Implement Linear Regression for predicting continuous values. |
| 7 | Support Vector Regression (SVR) | Implement Support Vector Machine-based regression. |
| 8 | Regularization | Apply regularization techniques to reduce overfitting. |
| 9 | CCTV Object Detection | Detect objects in CCTV footage using YOLO and OpenCV. |
| 10 | Multimodal Fake News Detection | Demonstrate fake news classification using textual and numerical features. |

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Processing:** NumPy, Pandas
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Computer Vision:** OpenCV
- **Object Detection:** Ultralytics YOLO
- **Development Environment:** Visual Studio Code

## 📂 Repository Structure

```text
AI-ML_Practicals/
│
├── practical1.py
├── practical2.py
├── practical3.py
├── practical4.py
├── practical5.py
├── practical6.py
├── practical7.py
├── practical8.py
├── practical9.py
├── practical10.py
│
├── .gitignore
└── README.md
```

**Note:** Practical 9 requires a CCTV video file named `cctv.mp4` in the appropriate working directory. The YOLO model weights may be downloaded automatically when the model is first initialized, depending on the environment and network availability.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/SOMESHSINGH25/AI-ML_Practicals.git
cd AI-ML_Practicals
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

### 4. Install the required libraries

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn opencv-python ultralytics
```

## ▶️ How to Run

Run each practical independently from the repository directory.

For example, to run Practical 1:

```bash
python practical1.py
```

To run Practical 2:

```bash
python practical2.py
```

Replace the filename with the practical you want to execute. Some practicals may generate plots, display video frames, or require additional input files.

## 💻 System Requirements

- Python 3.10 or later
- Windows, Linux, or macOS
- Visual Studio Code or another Python-compatible editor
- Internet connection for initial package installation and YOLO model download
- A CPU is sufficient for the provided implementations; a dedicated GPU is not required

## 📚 Learning Outcomes

By completing these practicals, learners can:

- Analyze relationships between variables.
- Understand supervised and unsupervised learning.
- Reduce data dimensionality using PCA.
- Implement classification and regression algorithms.
- Understand regularization and overfitting.
- Apply computer vision techniques for object detection.
- Explore text and numerical features for fake news classification.
- Practice implementing machine learning workflows using Python.

## ⚠️ Notes and Limitations

- These practicals are intended for educational purposes.
- Results depend on the datasets, model settings, and input quality.
- The multimodal fake news detection implementation is a demonstration and should not be treated as a reliable real-world fact-checking system without suitable datasets and evaluation.
- CCTV object detection accuracy depends on the video quality, lighting, camera angle, and objects present in the footage.

## 👨‍💻 Author

**Somesh Singh**  
B.Tech in Information Technology  
Bharati Vidyapeeth (Deemed to be University), College of Engineering, Pune

- GitHub: [SOMESHSINGH25](https://github.com/SOMESHSINGH25)

---

If you find this repository useful for learning AI/ML concepts, feel free to explore the implementations and build upon them.
