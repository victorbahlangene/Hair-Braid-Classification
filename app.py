from pathlib import Path
from fastai.vision.all import *
from fastai.vision.widgets import *

import streamlit as st
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


st.header("Hair Braid Classification")

# Load Model #
#path = Path()
learn_inf = load_learner(Path()/'hair.pkl')

# File uploader #
uploaded_file = st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    image_file = PILImage.create((uploaded_file))

# Display uploaded image #
st.image(image_file.to_thumb(400, 400), caption='Uploaded Image')

# make prediction and display result #
if st.button('Classify'):
    pred, pred_idx, probs = learn_inf.predict(image_file)
    st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
else:
    st.write(f'Click the button to classify')
