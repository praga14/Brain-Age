import streamlit as st
import nibabel as nib
import numpy as np
import pickle
import os
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import base64

# Load the models
with open("E:/Brain Age/cnn_model.pkl", "rb") as f:
    cnn_model = pickle.load(f)

with open("E:/Brain Age/model_resnet.pkl", "rb") as f:
    resnet_model = pickle.load(f)

# Preprocessing function for .nii file
def preprocess_nii(filepath, target_shape=(128, 128, 128)):
    img = nib.load(filepath)
    data = img.get_fdata()
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    from scipy.ndimage import zoom
    zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
    resized = zoom(data, zoom_factors, order=1)
    return resized.astype(np.float32).reshape(1, *target_shape, 1)

# Predict and compare
def predict_age(preprocessed_mri, actual_age):
    pred_cnn = cnn_model.predict(preprocessed_mri)[0][0]
    pred_resnet = resnet_model.predict(preprocessed_mri)[0][0]
    return [
        {"Model Name": "CNN", "Given Age": actual_age, "Predicted Age": pred_cnn, "Difference": abs(actual_age - pred_cnn)},
        {"Model Name": "ResNet", "Given Age": actual_age, "Predicted Age": pred_resnet, "Difference": abs(actual_age - pred_resnet)},
    ]

# Set background image (update the path as needed)
def set_bg_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

#  Main App Starts Here 

st.set_page_config(page_title="Brain Age Prediction", layout="centered")
set_bg_image("C:/Users/Pragatheesh/Downloads/WhatsApp Image 2025-05-15 at 18.36.49_907d0c05.jpg")


# Session state to simulate navigation
if "page" not in st.session_state:
    st.session_state.page = "main"

# Page 1: Upload & Input 
if st.session_state.page == "main":
    st.title("🧠 Brain Age Prediction Portal")
    st.write("Upload a T1-weighted MRI (.nii) and input actual age to predict brain age using CNN and ResNet.")

    uploaded_file = st.file_uploader("Upload .nii file", type=["nii"])
    given_age = st.number_input("Enter actual age", min_value=0, max_value=120, value=30)
    submit = st.button("Submit")

    if submit and uploaded_file is not None:
        filepath = os.path.join("temp_upload.nii")
        with open(filepath, "wb") as out:
            out.write(uploaded_file.read())

        try:
            with st.spinner("Processing and predicting..."):
                mri_data = preprocess_nii(filepath)
                predictions = predict_age(mri_data, given_age)
                st.session_state.predictions = predictions
                st.session_state.page = "results"
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.info("Please upload a .nii file and enter the age to proceed.")

# Page 2: Results
elif st.session_state.page == "results":
    st.title("📋 Prediction Results")

    pred_data = st.session_state.predictions
    st.dataframe(pred_data)

    # Scatter plot: Model vs Predicted Age
    st.subheader("📈 Model Performance - Scatter Plot")
    fig, ax = plt.subplots()
    models = [row["Model Name"] for row in pred_data]
    preds = [row["Predicted Age"] for row in pred_data]
    diffs = [row["Difference"] for row in pred_data]
    actual = pred_data[0]["Given Age"]

    ax.scatter(models, preds, color="purple", label="Predicted Age", s=100)
    ax.axhline(y=actual, color="gray", linestyle="--", label="Actual Age")
    ax.set_ylabel("Age")
    ax.set_title("Model Prediction Comparison")
    ax.legend()
    st.pyplot(fig)

    if st.button("🔙 Go Back"):
        st.session_state.page = "main"
        st.experimental_rerun()
