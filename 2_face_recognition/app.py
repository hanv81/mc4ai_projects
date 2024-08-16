import streamlit as st
import numpy as np
import pandas as pd
import glob
import time
import cv2
from deepface import DeepFace
from numpy.linalg import norm
from datetime import datetime

def main():
    st.title('Face Recognition')
    tab1, tab2, tab3 = st.tabs(['Face Register', 'Face Verify', 'Time Log'])
    with tab1:
        face_register()
    with tab2:
        face_verify()
    with tab3:
        df = pd.read_csv('data.csv', index_col=None)
        st.dataframe(df)

def face_register():
    name = st.text_input('Input Name')
    buffer = st.camera_input('Take a picture', key=1)
    if buffer is not None:
        if name == '':
            st.error('Please input name')
        else:
            with st.spinner('Please wait...'):
                bytes_data = buffer.getvalue()
                img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                faces = DeepFace.extract_faces(img)
                if len(faces) == 0:
                    st.error('Face not found')
                elif len(faces) > 1:
                    st.error('Only one face accepted')
                else:
                    face = np.uint8(faces[0]['face']*255)
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f'faces/{name}.png', face)
                    st.success('Register Done')

def face_verify():
    buffer = st.camera_input('Take a picture', key=2)
    if buffer is not None:
        with st.spinner('Please wait...'):
            t = time.time()
            bytes_data = buffer.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            embs = DeepFace.represent(img)
            reg_embs = load_register_face_embbedings()
            for emb in embs:
                for name, reg_emb in reg_embs:
                    similarity = 100 * cosine_similarity(emb['embedding'], reg_emb)
                    if similarity >= 70:
                        st.success(f'{name}: {round(similarity)}%')
                        data = {'Name':[name], 'Time':[str(datetime.now())]}
                        df = pd.DataFrame(data)
                        df.to_csv('data.csv', index=None)
                        break
            st.write(f'Process time: {round(time.time() - t, 3)}s')

@st.cache_data
def load_register_face_embbedings():
    return [(file_path[6:-4], DeepFace.represent(file_path, enforce_detection=False)[0]['embedding'])
            for file_path in glob.glob('faces/*.*')]

def cosine_similarity(v1, v2):
    return np.dot(v1,v2) / (norm(v1)*norm(v2))

main()