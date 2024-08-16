import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

def read_labels_input_shape():
    with open('labels_input_shape.txt') as f:
        labels = f.readline().split()
        input_shape = list(map(int, f.readline().split()))
    return labels, input_shape

def inference_image(model, labels, img, input_shape, draw_img=True):
    probs = model.predict(img.reshape(-1,*input_shape)).squeeze()*100
    ids = np.argsort(probs)[::-1]
    if draw_img:st.image(img)
    for i in ids[:5]:
        st.write(labels[i], ':', probs[i].round(decimals=2), '%')

@st.cache_data
def load_model_from_file():
    return load_model('model.h5')

def inference():
    labels, input_shape = read_labels_input_shape()
    st.subheader('Draw a letter (A-Z) or upload an image')
    col1, col2 = st.columns(2)
    with col1:
        canvas_result = st_canvas(
            stroke_width=15,
            stroke_color='rgb(255, 255, 255)',
            background_color='rgb(0, 0, 0)',
            height=150,
            width=150,
            key="canvas",
        )
    if canvas_result.image_data is not None:img = Image.fromarray(canvas_result.image_data)

    with col2:
        uploaded_file = st.file_uploader('', type=['png','jpg','bmp'])
        if uploaded_file is not None:img = Image.open(uploaded_file)

    if st.button('Predict'):
        model = st.session_state.get('model')
        try:
            if model is None:
                print('Load model from file')
                model = load_model('model.h5')
            img = img.resize(input_shape).convert('L')
            img = np.array(img, dtype=float)/255
            if uploaded_file is not None: 
                col3, col4, col5 = st.columns(3)
                with col3:
                    st.text('Original Image')
                    st.image(Image.open(uploaded_file))
                with col4:
                    st.text('Grayscale Image')
                    inference_image(model, labels, img, input_shape)
                with col5:
                    st.text('Invert Grayscale Image')
                    inference_image(model, labels, 1-img, input_shape)
            else:
                inference_image(model, labels, img, input_shape, False)
        except OSError:
            st.error('Model not found')

def main():
    st.set_page_config(
        page_title="Handwriting Recognition",
        page_icon="🏡",
    )

    inference()

main()