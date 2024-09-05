import os, faiss, cv2
import streamlit as st
import pandas as pd
from uform import get_model, Modality

index = faiss.read_index('database.index')
index_desc = faiss.read_index('description.index')
processors, models = get_model('unum-cloud/uform3-image-text-english-small')
model_image = models[Modality.IMAGE_ENCODER]
model_text = models[Modality.TEXT_ENCODER]
processor_image = processors[Modality.IMAGE_ENCODER]
processor_text = processors[Modality.TEXT_ENCODER]
df = pd.read_csv('database.csv', index_col=None)
df_movie = pd.read_csv('movie.csv', index_col=None)

@st.cache_data
def video_search(query_text):
    k = 3
    _, embedding = model_text.encode(processor_text(query_text))
    _, I = index.search(embedding, k)
    return df.loc[I[0]]

@st.cache_data
def text_search(query_text):
    k = 3
    _, embedding = model_text.encode(processor_text(query_text))
    _, I = index_desc.search(embedding, k)
    return df_movie.loc[I[0]]

def main():
    st.title('MOVIE RECOMMENDER SYSTEM')
    
    col1, col2 = st.columns(2)
    with col1:
        query_text = st.text_area('Search Content')
    with col2:
        searchby = st.radio('Search In', options=('Video', 'Description'))

    if len(query_text) > 0:
        result = video_search(query_text) if searchby == 'Video' else text_search(query_text)
        for i in result.index:
            row = result.loc[i]
            name, category = row[['Name', 'Category']]
            start_time = 0
            path = os.path.join('data', category, name, 'video.mp4')
            if searchby == 'Video':
                frame_id = row['Frame ID']
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                start_time = int(frame_id/fps)
                cap.release()

            st.subheader(name)
            col1, col2 = st.columns(2)
            with col1:
                thumb_path = os.path.join('data', category, name, 'thumbnail.jpg')
                st.image(thumb_path)

            with col2:
                st.video(path, start_time=start_time)
                desc_path = os.path.join('data', category, name, 'desc.txt')
                with open(desc_path) as f:
                    st.caption(f.read())

            st.write('\n\n')

main()