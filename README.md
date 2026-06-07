# Diabetic Retinopathy AI Diagnostic System

This system comprises a complete pipeline: downloading the dataset, training a Deep Learning model (MobileNetV2), extracting OCR text from blood reports, detecting symptoms, generating Grad-CAM explainable AI visualizations, and displaying it all on a Streamlit dashboard.

## Phase 3 Instructions: Dataset Setup
1. Go to Kaggle: [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
2. Download the dataset.
3. Extract the `train_images` into `dataset/` directory so that Keras `flow_from_directory` can read them. Ensure the subdirectories correspond to the class labels (0, 1, 2, 3, 4). You may need to write a quick script to organize them into subfolders based on `train.csv`.

## Local Execution
1. Install dependencies: `pip install -r requirements.txt` (Ensure Tesseract OCR is installed on your OS for `pytesseract`)
2. Run Training: `python training/train_model.py` (This will save `dr_model.h5` to the `app/` folder)
3. Run App: `cd app` and then `streamlit run app.py`

## Phase 14 Application Deployment (Hugging Face)
To deploy this project as a working AI website on Hugging Face Spaces:
1. **Login** to your Hugging Face account at [Hugging Face Spaces](https://huggingface.co/spaces).
2. Click **New Space**.
3. Choose **Streamlit** as the Space SDK.
4. Upload the following files to your space repository:
   - `app/app.py` -> `app.py`
   - `app/dr_model.h5` -> `dr_model.h5`
   - `requirements.txt`
   - `app/blood_report.py` -> `blood_report.py`
   - `app/symptoms.py` -> `symptoms.py`
   - `app/gradcam.py` -> `gradcam.py`
   *(Be sure to update imports in `app.py` if moved to root directory)*
5. Wait for the environment to build, and your app will be online!
