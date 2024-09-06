import faiss
import os,cv2
import pandas as pd
import numpy as np
from PIL import Image
from uform import get_model, Modality
from tqdm import tqdm

processors, models = get_model('unum-cloud/uform3-image-text-english-small')
model_image = models[Modality.IMAGE_ENCODER]
model_text = models[Modality.TEXT_ENCODER]
processor_image = processors[Modality.IMAGE_ENCODER]
processor_text = processors[Modality.TEXT_ENCODER]
data_path = 'data'

def create_index(embs, filename):
    embs = np.array(embs)
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, filename)

def create_database():
    embs, embs_desc = [], []
    df = pd.DataFrame(columns = ['Frame ID', 'Name', 'Category'])
    df_desc = pd.DataFrame(columns = ['Name', 'Category', 'Description'])
    
    for category in tqdm(os.listdir(data_path)):
        path = os.path.join(data_path, category)
        # print(category, os.listdir(path))
        for movie in tqdm(os.listdir(path)):
            desc_path = os.path.join('data', category, movie, 'desc.txt')
            with open(desc_path) as f:
                desc = f.read()
                df_desc.loc[df_desc.shape[0]] = [movie, category, desc]
                _, embedding = model_text.encode(processor_text(desc), return_features=True)
                embs_desc.append(embedding.flatten())

            movie_path = os.path.join(path, movie, 'video.mp4')
            cap = cv2.VideoCapture(movie_path)
            frame_id = 0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f'\nProcessing {movie}')
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # cv2.imshow('frame', frame)

                image_data = processor_image(Image.fromarray(frame))
                _, embedding = model_image.encode(image_data, return_features=True)
                embs.append(embedding.flatten())
                df.loc[len(df.index)] = [frame_id, movie, category]

                frame_id += 1
                if frame_id % 100 == 0:
                    print(f'frame {frame_id}/{frame_count}')

            cap.release()
            cv2.destroyAllWindows()

    create_index(embs, 'database.index')
    create_index(embs_desc, 'description.index')
    df.to_csv('database.csv', index=None)
    df_desc.to_csv('movie.csv', index=None)

create_database()