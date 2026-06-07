# Diabetic Retinopathy Detection & Risk Assessment System

AI-powered healthcare application for automated diabetic retinopathy screening using retinal fundus images, integrated with blood report analysis, symptom-based assessment, and explainable AI visualizations.

## Live Demo

**Hugging Face Space**
https://huggingface.co/spaces/Srindhi/dr-retinopathy-ai

**GitHub Repository**
https://github.com/davulurisrinidhi1/dr-retinopathy-ai

---

## Overview

Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness among diabetic patients. Early diagnosis is essential for timely treatment and reducing vision loss.

This project combines Deep Learning, Computer Vision, OCR, and Explainable AI techniques to assist in diabetic retinopathy screening. The system analyzes retinal fundus images, evaluates clinical indicators from blood reports, and provides visual explanations for model predictions.

---

## Features

### Retinal Image Classification

* Automated diabetic retinopathy severity prediction
* Multi-class DR grading
* Retinal image preprocessing using CLAHE enhancement
* Deep learning-based feature extraction

### Explainable AI

* Grad-CAM heatmap generation
* Visual interpretation of prediction results
* Identification of influential retinal regions

### Blood Report Analysis

* OCR-based extraction of HbA1c values
* Diabetes risk assessment
* Clinical parameter interpretation

### Symptom-Based Assessment

* Patient symptom evaluation
* Risk indication support
* Additional screening insights

### Interactive Web Application

* Built using Streamlit
* Upload retinal images directly through the browser
* Instant predictions and visual outputs

---

## Model Details

| Component           | Technology         |
| ------------------- | ------------------ |
| Deep Learning Model | MobileNetV2        |
| Image Enhancement   | CLAHE              |
| Explainability      | Grad-CAM           |
| OCR Engine          | EasyOCR            |
| Framework           | TensorFlow / Keras |
| Frontend            | Streamlit          |

---

## Dataset

The model was trained using the **APTOS 2019 Blindness Detection Dataset**, consisting of retinal fundus images categorized across multiple diabetic retinopathy severity levels.

The dataset includes:

* No DR
* Mild DR
* Moderate DR
* Severe DR
* Proliferative DR

---

## Performance

| Metric                         | Score  |
| ------------------------------ | ------ |
| Accuracy                       | 85.27% |
| Quadratic Weighted Kappa (QWK) | 0.887  |

These results indicate strong agreement between model predictions and clinical grading standards.

---

## Technology Stack

### Languages

* Python

### Machine Learning & AI

* TensorFlow
* Keras
* MobileNetV2

### Computer Vision

* OpenCV
* CLAHE
* Grad-CAM

### OCR & Data Processing

* EasyOCR
* NumPy
* Pandas

### Web Application

* Streamlit

### Visualization

* Matplotlib

---

## Project Structure

```text
dr-retinopathy-ai/
│
├── app/
│   ├── app.py
│   ├── blood_report.py
│   ├── gradcam.py
│   ├── symptoms.py
│   └── dr_model.h5
│
├── training/
│   ├── preprocess.py
│   ├── augmentation.py
│   └── train_model.py
│
├── requirements.txt
├── README.md
└── deploy_hf.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/davulurisrinidhi1/dr-retinopathy-ai.git
cd dr-retinopathy-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run app/app.py
```

---

## Future Improvements

* Ensemble-based prediction models
* PDF clinical report generation
* Multi-disease retinal screening
* Cloud-based patient management
* Enhanced model accuracy through larger datasets

---

## Disclaimer

This project is intended for educational, research, and screening purposes only. It should not be considered a substitute for professional medical diagnosis, treatment, or clinical decision-making.

---

## Author

**Davuluri Srinidhi**

Computer Science Engineering Student

LinkedIn: https://www.linkedin.com/in/dsvnssrinidhi01/

GitHub: https://github.com/davulurisrinidhi1
