import streamlit as st
import numpy as np
from zipfile import ZipFile
from PIL import Image
from uform import get_model, Modality
from numpy.linalg import norm
from deepface import DeepFace
from stqdm import stqdm
import cv2

processors, models = get_model('unum-cloud/uform3-image-text-english-base')
model_image = models[Modality.IMAGE_ENCODER]
model_text = models[Modality.TEXT_ENCODER]
processor_image = processors[Modality.IMAGE_ENCODER]
processor_text = processors[Modality.TEXT_ENCODER]

def normalize(a):
    n = norm(a, axis=1)
    return a / n.reshape(-1,1)

@st.cache_data
def read_zip_file(uploaded_file):
    with st.expander('Source Images'):
        cols = st.columns(4)
        with ZipFile(uploaded_file, 'r') as zip:
            imgs, embs = [], []
            for i, filename in enumerate(zip.namelist()):
                if filename.endswith('png') or filename.endswith('jpg') or filename.endswith('jpeg'):
                    with zip.open(filename) as f:
                        img = Image.open(f)
                        image_data = processor_image(img)
                        _, embedding = model_image.encode(image_data, return_features=True)
                        imgs.append(img)
                        embs.append(embedding.flatten()/norm(embedding))
                        with cols[i%4]:
                            st.image(img, f'{filename[-20:-4]}')

    return imgs, np.array(embs)

@st.cache_data
def get_image_embbeddings(img):
    try:
        embs = DeepFace.represent(img)
        return normalize(np.array([e['embedding'] for e in embs]))
    except:
        return None

def face_search(src_imgs):
    embs = None
    col1, col2 = st.columns(2)
    with col1:
        img_file = st.camera_input("📷 Take a picture")
        if img_file is not None:
            bytes_data = img_file.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            embs = get_image_embbeddings(img)
            if embs is None:
                st.warning('Face not found')

    with col2:
        img_file = st.file_uploader('🔼 Upload Image', type=['png','jpg','jpeg'])
        if img_file is not None:
            bytes_data = img_file.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            st.image(img_file)
            embs = get_image_embbeddings(img)
            if embs is None:
                st.info('Face not found')
    min_cosine = st.slider('Level of Similarity (%)', value=70, min_value=10, max_value=99, step=5)

    if embs is not None:
        result = []
        for i in stqdm(range(len(src_imgs))):
            src_embs = get_image_embbeddings(np.array(src_imgs[i]))
            if src_embs is not None:
                cosine = (src_embs @ embs.T).max()*100
                if cosine >= min_cosine:
                    result.append((cosine, i))
                    # st.image(src_imgs[i], caption=f'{round(cosine)}%')
        if result == []:
            st.info('Not found')
        else:
            result.sort(reverse=True)
            for cosine,i in result:
                st.image(src_imgs[i], caption=f'{round(cosine)}%')

def text_search(src_imgs, src_embs):
    col1, col2 = st.columns(2)
    with col1:
        text = st.text_area('Image Description')
    with col2:
        min_cosine = st.slider('Level of Similarity (%)', value=30, min_value=10, max_value=100, step=5)

    if len(text) > 0:
        text_data = processor_text(text)
        _, text_embedding = model_text.encode(text_data, return_features=True)
        text_embedding = text_embedding.flatten()/norm(text_embedding)
        cosine = (src_embs @ text_embedding)*100
        ids = np.where(cosine >= min_cosine)[0]
        if len(ids) == 0:
            st.info('Not found')
        else:
            result = []
            for i in ids:
                result.append((cosine[i], i))

            j = 0
            cols = st.columns(3)
            result.sort(reverse=True)
            st.success('Result')
            for cosine,i in result:
                with cols[j%3]:
                    st.image(src_imgs[i], f'{round(cosine)}%')
                j += 1

def main():
    st.set_page_config(page_title="Image Search", page_icon="🔍")
    st.title('🖼️ Image Search')
    with st.sidebar:
        uploaded_file = st.file_uploader('Upload Images', type=['zip'])
    if uploaded_file is not None:
        src_imgs, src_embs = read_zip_file(uploaded_file)

        tab1, tab2 = st.tabs(['🙂 Face Search', '📄 Text Search'])
        with tab1:
            face_search(src_imgs)
        with tab2:
            text_search(src_imgs, src_embs)
    else:
        st.text('👈 Please upload your images')

main()