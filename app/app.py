import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from blood_report import analyze_blood_report
from symptoms import get_symptoms_for_stage
from gradcam import generate_gradcam_heatmap, overlay_heatmap

# Ensure page configuration gives a professional, wide layout
st.set_page_config(page_title="AI Retinopathy Analyzer", layout="wide", page_icon="👁️")

# Custom CSS for a professional "doctor vibe"
st.markdown("""
<style>
    .main {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Load the model with caching so it doesn't reload on every UI interaction
@st.cache_resource
def load_dr_model():
    try:
        model = tf.keras.models.load_model("dr_model.h5")
        return model
    except Exception as e:
        return None

model = load_dr_model()

st.title("🥼 Diabetic Retinopathy Diagnostic System")
st.markdown("**Welcome to the AI-Assisted Retinopathy Analysis Dashboard.** Please upload patient retina scans and optional blood reports to generate a comprehensive diagnostic summary.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("Patient Inputs")
    retina_file = st.file_uploader("Upload Retina Image (Fundus Scan)", type=["jpg", "png", "jpeg"])
    blood_file = st.file_uploader("Upload Blood Sugar Report (Optional)", type=["jpg", "png", "jpeg"])
    
    analyze_btn = st.button("Generate AI Diagnostic Report", type="primary", use_container_width=True)

with col2:
    st.header("Diagnostic Dashboard")
    
    if analyze_btn:
        if not model:
            st.error("Model 'dr_model.h5' not found. Please ensure you have trained the model first.")
        elif not retina_file:
            st.warning("Please upload a retina image to begin the analysis.")
        else:
            with st.spinner("Analyzing retinal scan and patient records..."):
                # Process Image
                img = Image.open(retina_file).convert('RGB')
                st.image(img, caption="Original Fundus Scan", use_container_width=True)
                
                # Enhanced Retina Vessels
                st.subheader("Enhanced Retina Vessels")
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0)
                enhanced = clahe.apply(gray)
                st.image(enhanced, caption="CLAHE Enhanced Vessels", use_container_width=True)
                
                # Preprocess for model
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized)
                img_norm = img_array / 255.0
                img_batch = np.expand_dims(img_norm, 0)
                
                # Predict
                prediction = model.predict(img_batch)
                stage_idx = int(np.argmax(prediction[0]))
                confidence = float(np.max(prediction[0])) * 100
                
                classes = [
                    "No DR",
                    "Mild Non-Proliferative DR",
                    "Moderate Non-Proliferative DR",
                    "Severe Non-Proliferative DR",
                    "Proliferative DR"
                ]
                
                detected_state = classes[stage_idx]
                
                st.markdown(f"**Detected Stage:** {detected_state}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                
                # Prediction Probabilities
                st.subheader("Prediction Probabilities")
                prob_df = pd.DataFrame(
                    {"Probability": prediction[0]},
                    index=classes
                )
                st.bar_chart(prob_df)
                
                # Summary and recommendation based on stage
                st.subheader("Medical Analysis & Doctor's Advice")
                if stage_idx == 0:
                    st.success("✅ **STATUS: CLEAR**\n\nNo observable diabetic retinopathy detected. \n\n**Doctor's Recommendation**: Routine annual check-up recommended. Keep blood sugar maintained.")
                    visit_urgency = "Low - Annual visits"
                    home_remedies = "Maintain standard healthy diet and exercise."
                elif stage_idx in [1, 2]:
                    st.warning("⚠️ **STATUS: CAUTION (Early to Moderate DR)**\n\nEarly to moderate retinal damage detected (Microaneurysms, possible exudates).\n\n**Doctor's Recommendation**: Schedule a follow-up with a local ophthalmologist within **3-6 months**.")
                    visit_urgency = "Moderate - See ophthalmologist within 3-6 months"
                    home_remedies = "Strict glycemic control (HbA1c target < 7%), blood pressure management, quit smoking, increase omega-3 intake."
                else:
                    st.error("🚨 **STATUS: CRITICAL (Advanced DR)**\n\nAdvanced retinal damage detected (Significant hemorrhages, neovascularization).\n\n**Doctor's Recommendation**: Seek specialist care **IMMEDIATELY**. Potential intervention required (Laser Photocoagulation / Anti-VEGF injections).")
                    visit_urgency = "VERY HIGH - Immediate specialist intervention required"
                    home_remedies = "Medical intervention is mandatory. While waiting, ensure absolute strict blood sugar and blood pressure control. Do not strain eyes."
                
                # Grad-CAM specific
                st.subheader("Disease Heatmap (Grad-CAM)")
                try:
                    heatmap = generate_gradcam_heatmap(model, img_batch, stage_idx)
                    cam_img = overlay_heatmap(img_array, heatmap)
                    st.image(cam_img, caption="Grad-CAM Hotspots (Areas of concern)", use_container_width=True)
                except Exception as e:
                    st.warning("Could not generate Grad-CAM visualization. (Check tf-keras-vis compatibility)")

                # Generate Report
                st.markdown("---")
                st.subheader("Automated Medical Report")
                
                report_md = f"### 🩺 Comprehensive Pathology Summary\n\n"
                report_md += f"**Primary AI Diagnosis:** {detected_state} (Confidence: {confidence:.1f}%)\n\n"
                report_md += f"**Ophthalmologist Visit Urgency:** {visit_urgency}\n\n"
                
                # Symptoms
                symptoms = get_symptoms_for_stage(stage_idx)
                if symptoms:
                    report_md += "#### ⚠️ Inferable Symptoms (Patient may be experiencing):\n"
                    for s in symptoms:
                        report_md += f"- {s}\n"
                else:
                    report_md += "#### ⚠️ Inferable Symptoms:\n- None or asymptomatic.\n"
                
                report_md += "\n#### 💊 Recommended Precautions & Home Remedies:\n"
                report_md += f"- {home_remedies}\n\n"
                
                # Blood Report OCR
                if blood_file:
                    blood_img = Image.open(blood_file)
                    blood_analysis = analyze_blood_report(blood_img)
                    report_md += f"#### 🩸 Blood Report Integrations:\n{blood_analysis}"
                else:
                    report_md += "#### 🩸 Blood Report Integrations:\n- No blood report provided. Unable to correlate with HbA1c."
                
                st.info(report_md)
                
                st.balloons()
