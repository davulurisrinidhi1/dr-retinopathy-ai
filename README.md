# Diabetic Retinopathy Detection & Risk Assessment System

An AI-powered healthcare screening application that analyzes retinal fundus images for diabetic retinopathy detection, integrates blood report analysis through OCR, performs symptom-based risk assessment, and provides visual explanations using Grad-CAM.

## Live Demo

🔗 **Hugging Face Deployment**
https://huggingface.co/spaces/Srindhi/dr-retinopathy-ai

🔗 **GitHub Repository**
https://github.com/davulurisrinidhi1/dr-retinopathy-ai

---

## Project Overview

Diabetic Retinopathy (DR) is a diabetes-related eye disease that can lead to severe vision impairment if not detected early.

This project combines Deep Learning, Computer Vision, OCR, and Explainable AI techniques to build an interactive screening platform capable of:

* Analyzing retinal fundus images
* Assessing diabetic retinopathy severity
* Extracting HbA1c values from blood reports
* Evaluating patient symptoms
* Providing visual explanations for model predictions

The system is designed as a proof-of-concept healthcare application demonstrating the practical integration of AI technologies in medical image analysis.

---

## Key Features

### Retinal Fundus Image Analysis

* Deep learning-based retinal image classification
* Multi-level diabetic retinopathy assessment
* Automated image preprocessing pipeline
* Instant prediction through a web interface

### Image Enhancement

* CLAHE (Contrast Limited Adaptive Histogram Equalization)
* Improved retinal feature visibility
* Enhanced image quality before model inference

### Explainable AI

* Grad-CAM visualization
* Highlights image regions influencing predictions
* Improves model interpretability and transparency

### Blood Report Analysis

* OCR-based extraction of HbA1c values
* Automatic clinical interpretation
* Additional diabetes risk insights

### Symptom Assessment

* Symptom-driven risk evaluation
* Supportive healthcare screening information

### Interactive Web Application

* Built with Streamlit
* Easy image upload and prediction workflow
* User-friendly interface

---

## System Architecture

1. User uploads a retinal fundus image.
2. Image preprocessing is applied using CLAHE.
3. MobileNetV2-based model performs prediction.
4. Grad-CAM generates visual explanations.
5. Optional blood report analysis extracts HbA1c values.
6. Symptom inputs provide additional risk assessment.
7. Results are displayed through the Streamlit interface.

---

## Technologies Used

### Programming Language

* Python

### Machine Learning & Deep Learning

* TensorFlow
* Keras
* MobileNetV2

### Computer Vision

* OpenCV
* CLAHE
* Grad-CAM

### OCR

* EasyOCR

### Data Processing

* NumPy
* Pandas

### Web Framework

* Streamlit

### Visualization

* Matplotlib

---

## Dataset

The model was developed using retinal fundus image datasets inspired by the APTOS 2019 Blindness Detection challenge.

The project focuses on demonstrating an end-to-end diabetic retinopathy screening workflow rather than producing a clinically validated diagnostic system.

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

Run the application:

```bash
streamlit run app/app.py
```

---

## Future Enhancements

* Improved model accuracy using larger datasets
* Ensemble learning approaches
* PDF medical report generation
* Multi-disease retinal screening
* Cloud-based patient record integration
* Advanced clinical decision-support features

---

## Disclaimer

This project is intended for educational, research, and demonstration purposes only. It should not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

---

## Author

**Davuluri Srinidhi**

Computer Science Engineering Student

LinkedIn: https://www.linkedin.com/in/dsvnssrinidhi01/

GitHub: https://github.com/davulurisrinidhi1
