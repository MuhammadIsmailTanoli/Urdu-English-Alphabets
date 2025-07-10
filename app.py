"""
Save this file as `app.py` and run it with:

  pip install -r requirements.txt
  streamlit run app.py

Note: Streamlit apps cannot be viewed inline in Jupyter; use `streamlit run` from a terminal.
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer, Conv2D

# Monkey-patch InputLayer.from_config to support 'batch_shape'
@classmethod
def _inputlayer_from_config(cls, config, custom_objects=None):
    if 'batch_shape' in config:
        config['batch_input_shape'] = config.pop('batch_shape')
    return cls(**config)
InputLayer.from_config = _inputlayer_from_config

# Monkey-patch Conv2D.from_config to handle 'dtype' dict
@classmethod
def _conv2d_from_config(cls, config, custom_objects=None):
    if 'dtype' in config and isinstance(config['dtype'], dict):
        # Extract the actual dtype string from the policy config
        config['dtype'] = config['dtype'].get('config', {}).get('name', None)
    return cls(**config)
Conv2D.from_config = _conv2d_from_config

# Load models once and cache
def load_models():
    tf.keras.backend.clear_session()
    urdu_model = load_model('Models/urdu_model.keras')
    tf.keras.backend.clear_session()
    english_model = load_model('Models/english_model.keras')
    return urdu_model, english_model

@st.cache_resource
def get_models():
    return load_models()

urdu_model, english_model = get_models()

# English letter mapping
def get_word_dict():
    return {i: chr(65 + i) for i in range(26)}
word_dict = get_word_dict()

# App title
st.title("Handwritten Letter Similarity Checker")

# Sidebar controls
language = st.sidebar.radio("Select Language:", ["English", "Urdu"])
target_input = st.sidebar.text_input("Target (Letter for English, Index for Urdu):")
stroke_width = st.sidebar.slider("Stroke Width:", 5, 30, 15)

# Set canvas colors
if language == "English":
    bg_color = "#000000"
    stroke_color = "#FFFFFF"
else:
    bg_color = "#000000"
    stroke_color = "#FFFFFF"

# Draw canvas (increased size)
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    width=560,
    height=560,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction logic
def predict_letter(image_array, lang, target):
    gray = cv2.cvtColor(image_array[:, :, :3].astype('uint8'), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (28, 28))
    norm = resized.astype('float32') / 255.0
    inp = norm.reshape(1, 28, 28, 1)

    sim_val = None
    if lang == "English":
        preds = english_model.predict(inp)
        idx = int(np.argmax(preds))
        letter = word_dict[idx]
        tgt = target.strip().upper()
        inv = {v:k for k,v in word_dict.items()}
        if tgt in inv:
            sim_val = preds[0][inv[tgt]] * 100
            sim = f"{sim_val:.2f}%"
        else:
            sim = "Invalid English target letter."
        return letter, sim, resized, sim_val
    else:
        try:
            ti = int(target)
            preds = urdu_model.predict(inp)[0]
            sim_val = preds[ti] * 100
            sim = f"{sim_val:.2f}%"
            return None, sim, resized, sim_val
        except:
            return None, "Invalid Urdu target index.", resized, None

# On button click
if st.button("Predict"):
    if canvas_result.image_data is None:
        st.warning("Please draw a letter on the canvas first.")
    else:
        letter, similarity_text, proc_img, sim_val = predict_letter(
            canvas_result.image_data,
            language,
            target_input
        )
        if language == "English":
            st.write(f"**Predicted Letter:** {letter}")
            st.write(f"**Similarity to '{target_input.strip().upper()}':** {similarity_text}")
        else:
            st.write(f"**Similarity:** {similarity_text}")

        if sim_val is not None:
            if sim_val >= 75:
                st.success("Great job! 🎉 Your handwriting is very similar to the target.")
            else:
                st.error("Try again! ❗ The similarity is below 75%.")

        st.image(proc_img, caption="Processed 28×28 Image", use_column_width=False)
