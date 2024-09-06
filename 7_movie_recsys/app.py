import os, faiss, cv2, random
import streamlit as st
import pandas as pd
from uform import get_model, Modality
from annotated_text import annotated_text

index = faiss.read_index('database.index')
index_desc = faiss.read_index('description.index')
processors, models = get_model('unum-cloud/uform3-image-text-english-small')
model_text = models[Modality.TEXT_ENCODER]
processor_text = processors[Modality.TEXT_ENCODER]
df = pd.read_csv('database.csv', index_col=None)
df_movie = pd.read_csv('movie.csv', index_col=None)
colors = "#8ef", "#faa", "#afa", "#fea", "#8ef", "#afa", "#faf", '#0fe', '#2ab', '#fc2', '#cf8', '#abc', '#cab', '#bca'

@st.cache_data
def search(query_text, _index, df):
    _, embedding = model_text.encode(processor_text(query_text))
    faiss.normalize_L2(embedding)
    D, I = _index.search(embedding, k=3)
    return df.loc[I[0]], D[0]

def group_similar_text(query_text, desc):
    desc_lower = desc.lower().split()
    flag = [word in query_text.lower().split() for word in desc_lower]
    group = []
    desc = desc.split()
    i = 0
    while i < len(flag):
        if not flag[i]:
            group.append(desc[i] + ' ')
            i += 1
        else:
            phrase = ''
            while i < len(flag) and flag[i]:
                phrase += desc[i] + ' '
                i += 1
            c = random.choice(colors)
            group.append((phrase, '', c))
    return group

def main():
    st.title('MOVIE RECOMMENDER SYSTEM')

    with st.expander('Movie Database'):
        st.dataframe(df_movie, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        query_text = st.text_area('Search Content')
    with col2:
        options = 'Video Frames', 'Video Description'
        searchby = options.index(st.radio('Search In', options=options))

    if len(query_text) > 0:
        result, D = search(query_text, index, df) if searchby == 0 else search(query_text, index_desc, df_movie)
        for i in range(len(result)):
            row = result.iloc[i]
            name, category = row[['Name', 'Category']]
            start_time = 0
            path = os.path.join('data', category, name, 'video.mp4')
            if searchby == 0:
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
                    desc = f.read()
                    if searchby == 0:
                        st.caption(desc)
                    else:
                        group = group_similar_text(query_text, desc)
                        annotated_text(group)
                st.write('\n\n')
                st.write('Similarity:', round((D[i])*100), '%')
            st.write('\n\n')

main()