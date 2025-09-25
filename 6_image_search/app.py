import cv2
import streamlit as st
import numpy as np
from zipfile import ZipFile
from PIL import Image
from uform import get_model, Modality
from numpy.linalg import norm
from deepface import DeepFace

processors, models = get_model('unum-cloud/uform3-image-text-english-base')
model_image = models[Modality.IMAGE_ENCODER]
model_text = models[Modality.TEXT_ENCODER]
processor_image = processors[Modality.IMAGE_ENCODER]
processor_text = processors[Modality.TEXT_ENCODER]
models = "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"

@st.cache_data
def normalize(a):
    n = norm(a, axis=1)
    return a / n.reshape(-1,1)

@st.cache_data
def read_zip_file(uploaded_file):
    cols = st.columns(4)
    with ZipFile(uploaded_file, 'r') as zip:
        imgs, embs = [], []
        for i, filename in enumerate(zip.namelist()):
            if filename.endswith('png') or filename.endswith('jpg') or filename.endswith('jpeg'):
                with zip.open(filename) as f:
                    img = Image.open(f)
                    image_data = processor_image(img)
                    _, embedding = model_image.encode(image_data)
                    imgs.append(img)
                    embs.append(embedding.flatten()/norm(embedding))
                    with cols[i%4]:
                        st.image(img, f'{filename[-20:-4]}')

    return imgs, np.array(embs)

@st.cache_data
def get_image_embbeddings(img, model_name):
    try:
        embs = DeepFace.represent(img, model_name=model_name)
        return normalize(np.array([e['embedding'] for e in embs]))
    except:
        return None

@st.cache_data
def represent_faces(img, model_name):
    try:
        return DeepFace.represent(img, model_name=model_name)
    except:
        return None

def face_search(src_imgs, model_name):
    embs = None
    col1, col2 = st.columns(2)
    with col1:
        img_file = st.camera_input("📷 Take a picture")
        if img_file is not None:
            bytes_data = img_file.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            embs = get_image_embbeddings(img, model_name)
            if embs is None:
                st.warning('Face not found', icon='⚠️')

    with col2:
        img_file = st.file_uploader('🔼 Upload Image', type=['png','jpg','jpeg'])
        if img_file is not None:
            bytes_data = img_file.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            embs = represent_faces(img, model_name)
            if embs is None:
                st.warning('Face not found', icon='⚠️')
            else:
                for i,emb in enumerate(embs, start=1):
                    x,y,w,h = emb['facial_area'].values()
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
                    cv2.putText(img, f'{i}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                embs = normalize(np.array([e['embedding'] for e in embs]))
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if embs is not None:
        min_cosine = st.slider('Level of Similarity (%)', value=70, min_value=10, max_value=99, step=5)
        result = []
        for i in range(len(src_imgs)):
            img = np.array(src_imgs[i])
            faces = represent_faces(img, model_name)
            if faces is not None:
                src_embs = normalize(np.array([f['embedding'] for f in faces]))
                cosine = np.einsum('ij,kj->ik', src_embs, embs)*100
                found = False
                for j,face in enumerate(faces):
                    if cosine[j].max() >= min_cosine:
                        found = True
                        k = cosine[j].argmax()
                        x,y,w,h = face['facial_area'].values()
                        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
                        cv2.putText(img, f'{k+1}:{round(cosine[j,k])}%', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                if found:
                    result.append(img)

        if result == []:
            st.warning('Not found', icon='⚠️')
        else:
            j = 0
            st.success('Result')
            cols = st.columns(2)
            for img in result:
                with cols[j]:
                    st.image(img)
                j = 1 - j

def text_search(src_imgs, src_embs):
    col1, col2 = st.columns(2)
    with col1:
        text = st.text_area('Image Description')
    with col2:
        min_cosine = st.slider('Level of Similarity (%)', value=30, min_value=10, max_value=100, step=5)

    if len(text) > 0:
        text_data = processor_text(text)
        _, text_embedding = model_text.encode(text_data)
        text_embedding = text_embedding.flatten()/norm(text_embedding)
        cosine = (src_embs @ text_embedding)*100
        ids = np.where(cosine >= min_cosine)[0]
        if len(ids) == 0:
            st.info('Not found')
        else:
            result = [(cosine[i], i) for i in ids]
            result.sort(reverse=True)

            j = 0
            st.success('Result')
            cols = st.columns(3)
            for cosine,i in result:
                with cols[j%3]:
                    st.image(src_imgs[i], f'{round(cosine)}%')
                j += 1

def main():
    st.set_page_config(page_title="Image Search", page_icon="🔍")
    st.title('🖼️ Image Search')
    with st.sidebar:
        uploaded_file = st.file_uploader('Upload Images', type=['zip'])

    if uploaded_file is None:
        st.text('👈 Please upload your images')
    else:
        with st.sidebar:
            model_name = st.selectbox('Select Model', options=models)
        with st.expander('Source Images'):
            src_imgs, src_embs = read_zip_file(uploaded_file)

        tab1, tab2 = st.tabs(['🙂 Face Search', '📄 Text Search'])
        with tab1:
            face_search(src_imgs, model_name)
        with tab2:
            text_search(src_imgs, src_embs)

main()