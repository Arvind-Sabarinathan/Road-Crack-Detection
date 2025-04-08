import streamlit as st

# 🛠️ Set page configuration FIRST
st.set_page_config(page_title="Asphalt Road Crack Detection", page_icon="🛣️", layout="wide")

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from huggingface_hub import hf_hub_download

# ---- Define Dice Coefficient & Loss ----
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

MODEL_PATH = hf_hub_download(
    repo_id="Arvind-Sabarinathan/crack_detection_unet",
    filename="crack_detection_unet.h5",
)
# ---- Load the trained U-Net model ----
@st.cache_resource
def load_unet_model():
    return load_model(MODEL_PATH, custom_objects={"dice_coef": dice_coef, "dice_loss": dice_loss})

unet = load_unet_model()

# ---- Streamlit UI ----
st.markdown("<h1 style='text-align: center;'>🛣️ Asphalt Road Crack Detection App</h1>", unsafe_allow_html=True)
st.write("### Upload one or multiple images, and the model will detect cracks.")

# ---- Sidebar Settings ----
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Segmentation Threshold", 0.1, 0.9, 0.3, step=0.05)
opacity = st.sidebar.slider("Heatmap Opacity", 0.1, 1.0, 0.5, step=0.1)
colormap_choice = st.sidebar.selectbox("Choose Heatmap Colormap", ["JET", "PLASMA", "VIRIDIS"], index=0)

# ---- Upload Images ----
uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

processed_images = []  # Store processed images & crack details

if uploaded_files:
    num_images = len(uploaded_files)
    st.write(f"✅ **{num_images} image(s) uploaded**")

    if st.button("🚀 Process Images"):
        st.write("### Results:")

        img_display_size = 300  # Optimal size for visibility
        colormaps = {"JET": cv2.COLORMAP_JET, "PLASMA": cv2.COLORMAP_PLASMA, "VIRIDIS": cv2.COLORMAP_VIRIDIS}
        colormap = colormaps[colormap_choice]

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            # Read the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Preprocess Image (Resize & Normalize)
            img_size = 256
            image_resized = cv2.resize(image, (img_size, img_size)) / 255.0  
            input_image = np.expand_dims(image_resized, axis=0)

            # Predict Crack Mask
            predicted_mask = unet.predict(input_image)[0]
            binary_mask = (predicted_mask > threshold).astype(np.uint8)

            # **Calculate Crack Area**
            total_pixels = img_size * img_size  # Total image size
            crack_pixels = np.sum(binary_mask)  # Count white pixels in mask
            crack_percentage = (crack_pixels / total_pixels) * 100  # Percentage calculation

            # Generate Heatmap Overlay
            heatmap = cv2.applyColorMap((predicted_mask.squeeze() * 255).astype(np.uint8), colormap)
            overlay = cv2.addWeighted(cv2.resize(image, (img_size, img_size)), 1 - opacity, heatmap, opacity, 0)

            # Convert images to bytes for report
            _, original_buffer = cv2.imencode(".png", cv2.resize(image, (img_size, img_size)))
            _, heatmap_buffer = cv2.imencode(".png", overlay)

            original_bytes = BytesIO(original_buffer.tobytes())
            heatmap_bytes = BytesIO(heatmap_buffer.tobytes())

            # Store processed data
            processed_images.append((uploaded_file.name, crack_pixels, crack_percentage, original_bytes, heatmap_bytes))

            # ---- Display Results ----
            st.write(f"### {idx}. {uploaded_file.name}")
            col1, col2, col3 = st.columns([2, 2, 1], gap="large")

            with col1:
                st.image(cv2.resize(image, (img_display_size, img_display_size)), caption="Original Image", width=img_display_size)
            with col2:
                st.image(cv2.resize(overlay, (img_display_size, img_display_size)), caption="Crack Heatmap", width=img_display_size)
            with col3:
                st.download_button(
                    label="📥 Download",
                    data=heatmap_bytes,
                    file_name=f"Crack_Heatmap_{idx}.png",
                    mime="image/png"
                )

            st.write(f"**Crack Area:** {crack_pixels} pixels")
            st.write(f"**Crack Percentage:** {crack_percentage:.2f}%")
            st.markdown("---")  # Divider for clarity

# ---- Generate PDF Report ----
def generate_pdf_report(processed_images):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(180, height - 50, "Asphalt Road Crack Detection Report")
    
    y_position = height - 100

    for idx, (file_name, crack_area, crack_percentage, original_img, heatmap_img) in enumerate(processed_images, start=1):
        pdf.setFont("Helvetica", 12)
        pdf.drawString(50, y_position, f"{idx}. {file_name}")
        pdf.drawString(50, y_position - 20, f"Crack Area: {crack_area} pixels")
        pdf.drawString(50, y_position - 40, f"Crack Percentage: {crack_percentage:.2f}%")

        # Add images
        original_reader = ImageReader(original_img)
        heatmap_reader = ImageReader(heatmap_img)

        pdf.drawImage(original_reader, 50, y_position - 200, width=200, height=150, preserveAspectRatio=True)
        pdf.drawImage(heatmap_reader, 300, y_position - 200, width=200, height=150, preserveAspectRatio=True)

        y_position -= 250  # Move down for next entry

        if y_position < 100:  # Add a new page if space is low
            pdf.showPage()
            y_position = height - 50

    pdf.save()
    buffer.seek(0)
    return buffer

# ---- Download Report Button ----
if processed_images:
    report_file = generate_pdf_report(processed_images)

    st.download_button(
        label="📄 Download Report",
        data=report_file,
        file_name="Crack_Detection_Report.pdf",
        mime="application/pdf"
    )
