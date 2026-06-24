import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Oman Plant Disease Detection", page_icon="🌿", layout="wide")
st.title("Plant Disease Detection for Omani Agriculture")
st.write("A simple Digital Image Processing prototype for detecting suspected disease areas in plant leaves, especially date palm leaves.")


def process_leaf(image_rgb):
    """Return leaf mask, disease mask, overlay image, and disease percentage."""
    # Resize for faster processing
    max_width = 900
    h, w = image_rgb.shape[:2]
    if w > max_width:
        scale = max_width / w
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))

    # OpenCV uses BGR, Streamlit/PIL uses RGB
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 1) Leaf mask: mostly green/yellow plant pixels
    lower_leaf = np.array([15, 30, 30])
    upper_leaf = np.array([95, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    # 2) Disease-like colors: yellow/brown/dark spots inside the leaf area
    yellow_brown = cv2.inRange(hsv, np.array([10, 45, 50]), np.array([40, 255, 255]))
    dark_spots = cv2.inRange(hsv, np.array([0, 20, 0]), np.array([180, 255, 85]))
    disease_mask = cv2.bitwise_or(yellow_brown, dark_spots)
    disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)

    # 3) Morphological processing to clean noise
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    leaf_pixels = int(np.count_nonzero(leaf_mask))
    disease_pixels = int(np.count_nonzero(disease_mask))
    percentage = (disease_pixels / leaf_pixels * 100) if leaf_pixels > 0 else 0

    if percentage < 3:
        label = "Healthy / Low suspected disease"
    elif percentage < 12:
        label = "Suspected Moderate disease"
    else:
        label = "Suspected High disease"

    overlay = image_rgb.copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255  # red channel in RGB
    overlay[disease_mask > 0] = cv2.addWeighted(overlay, 0.45, red, 0.55, 0)[disease_mask > 0]

    return leaf_mask, disease_mask, overlay, percentage, label

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    image_rgb = np.array(img)

    leaf_mask, disease_mask, overlay, percentage, label = process_leaf(image_rgb)

    st.subheader("Result")
    st.metric("Detected suspected disease area", f"{percentage:.2f}%")
    st.success(label)

    col1, col2, col3 = st.columns(3)
    col1.image(image_rgb, caption="Original image", use_container_width=True)
    col2.image(disease_mask, caption="Disease mask", use_container_width=True, clamp=True)
    col3.image(overlay, caption="Detected areas highlighted in red", use_container_width=True)

    st.info("This is a simple DIP prototype. It uses color thresholding and morphology, not a trained AI model.")
else:
    st.warning("Upload a clear image of a leaf to start. You can use sample_images/date_palm_leaf_sample.png.")
