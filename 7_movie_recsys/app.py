import os, faiss, cv2, random
import streamlit as st
import pandas as pd
from uform import get_model, Modality
from annotated_text import annotated_text

ROOT = '7_movie_recsys'
db_index_file_path = os.path.join(ROOT, 'database.index')
db_df_file_path = os.path.join(ROOT, 'database.csv')
desc_index_file_path = os.path.join(ROOT, 'description.index')
movie_df_file_path = os.path.join(ROOT, 'movie.csv')
colors = "#8ef", "#faa", "#afa", "#fea", "#8ef", "#afa", "#faf", '#0fe', '#2ab', '#fc2', '#cf8', '#abc', '#cab', '#bca'

@st.cache_resource
def load_data():
    with st.spinner('Loading Resource ...'):
        index = faiss.read_index(db_index_file_path)
        index_desc = faiss.read_index(desc_index_file_path)
        df = pd.read_csv(db_df_file_path, index_col=None)
        df_movie = pd.read_csv(movie_df_file_path, index_col=None)

        processors, models = get_model('unum-cloud/uform3-image-text-english-small')
        model = models[Modality.TEXT_ENCODER]
        processor = processors[Modality.TEXT_ENCODER]
    return index, index_desc, df, df_movie, model, processor

@st.cache_data
def search(query_text, _index, _model, _processor, df):
    _, embedding = _model.encode(_processor(query_text))
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
    index, index_desc, df, df_movie, model, processor = load_data()
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
        result, D = search(query_text, index, model, processor, df) if searchby == 0 else search(query_text, index_desc, model, processor, df_movie)
        for i in range(len(result)):
            row = result.iloc[i]
            name, category = row[['Name', 'Category']]
            start_time = 0
            path = os.path.join(ROOT, 'data', category, name, 'video.mp4')
            if searchby == 0:
                frame_id = row['Frame ID']
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                start_time = int(frame_id/fps)
                cap.release()

            st.subheader(name)
            col1, col2 = st.columns(2)
            with col1:
                thumb_path = os.path.join(ROOT, 'data', category, name, 'thumbnail.jpg')
                st.image(thumb_path)

            with col2:
                st.video(path, start_time=start_time)
                desc_path = os.path.join(ROOT, 'data', category, name, 'desc.txt')
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