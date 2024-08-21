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
            embs = normalize(np.array([e['embedding'] for e in embs]))
            registered_names, registered_embs = load_register_face_embbedings()
            cosine = embs @ registered_embs.T
            for i in range(len(embs)):
                j = cosine[i].argmax()
                if cosine[i,j] >= .7:
                    st.success(f'{registered_names[j]}: {round(cosine[i,j] * 100)}%')
                    data = {'Name':[registered_names[j]], 'Time':[str(datetime.now())]}
                    df = pd.DataFrame(data)
                    df.to_csv('data.csv', index=None)

            st.write(f'Process time: {round(time.time() - t, 3)}s')

@st.cache_data
def load_register_face_embbedings():
    names = [file_path[6:-4] for file_path in glob.glob('faces/*.*')]
    embs = [DeepFace.represent(file_path, enforce_detection=False)[0]['embedding'] for file_path in glob.glob('faces/*.*')]
    return names, normalize(np.array(embs))

def normalize(a):
    n = norm(a, axis=1)
    return a / n.reshape(-1,1)

main()