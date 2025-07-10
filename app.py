"""
Save this file as `app.py` and run it with:

  pip install streamlit streamlit-drawable-canvas opencv-python tensorflow
  streamlit run app.py

Note: Streamlit apps cannot be viewed inline in Jupyter; use `streamlit run` from a terminal.
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load models once and cache
@st.cache_resource
def load_models():
    # Clear any previous models to avoid name_scope issues
    tf.keras.backend.clear_session()
    urdu_model = load_model('Models/urdu_model.keras')
    tf.keras.backend.clear_session()
    english_model = load_model('Models/english_model.keras')
    return urdu_model, english_model

urdu_model, english_model = load_models()

# English letter mapping
def get_word_dict():
    return {i: chr(65 + i) for i in range(26)}
word_dict = get_word_dict()

# App title
st.title("Handwritten Letter Similarity Checker")

# Sidebar controls
language = st.sidebar.radio("Select Language:", ["English", "Urdu"])
target_input = st.sidebar.text_input("Target (Letter for English, Index for Urdu):")
stroke_width = 20

# Set canvas colors (swap for English to white-on-black)
if language == "English":
    bg_color     = "#000000"  # black background for English
    stroke_color = "#FFFFFF"  # white strokes for English
else:
    bg_color     = "#000000"  # black background for Urdu
    stroke_color = "#FFFFFF"  # white strokes for Urdu

# Draw canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Handle prediction
def predict_letter(image_array, lang, target):
    # Preprocess image
    gray = cv2.cvtColor(image_array[:, :, :3].astype('uint8'), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (28, 28))
    norm    = resized.astype('float32') / 255.0
    inp     = norm.reshape(1, 28, 28, 1)

    sim_val = None
    if lang == "English":
        preds = english_model.predict(inp)
        idx   = int(np.argmax(preds))
        letter= word_dict[idx]
        tgt   = target.strip().upper()
        inv   = {v:k for k,v in word_dict.items()}
        if tgt in inv:
            sim_val = preds[0][inv[tgt]] * 100
            sim = f"{sim_val:.2f}%"
        else:
            sim = "Invalid English target letter."
        return letter, sim, resized, sim_val
    else:
        try:
            ti    = int(target)
            preds = urdu_model.predict(inp)[0]
            sim_val = preds[ti] * 100
            sim = f"{sim_val:.2f}%"
            return None, sim, resized, sim_val
        except:
            return None, "Invalid Urdu target index.", resized, None

# On predict button click
if st.button("Predict"):
    if canvas_result.image_data is None:
        st.warning("Please draw a letter on the canvas first.")
    else:
        letter, similarity_text, proc_img, sim_val = predict_letter(
            canvas_result.image_data,
            language,
            target_input
        )
        # Display results
        if language == "English":
            st.write(f"**Predicted Letter:** {letter}")
            st.write(f"**Similarity to '{target_input.strip().upper()}':** {similarity_text}")
        else:
            st.write(f"**Similarity:** {similarity_text}")

        # Reward or failure message
        if sim_val is not None:
            if sim_val >= 75:
                st.success("Great job! 🎉 Your handwriting is very similar to the target.")
            else:
                st.error("Try again! ❗ The similarity is below 75%.")

        # Show processed image
        st.image(proc_img, caption="Processed 28×28 Image", use_column_width=False)
