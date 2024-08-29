import faiss
import os,cv2
import pandas as pd
import numpy as np
from PIL import Image
from uform import get_model, Modality
from tqdm import tqdm

processors, models = get_model('unum-cloud/uform3-image-text-english-small')
model_image = models[Modality.IMAGE_ENCODER]
processor_image = processors[Modality.IMAGE_ENCODER]
data_path = 'data'

def create_database():
    embs = []
    df = pd.DataFrame(columns = ['Frame ID', 'Name', 'Category'])
    
    for category in tqdm(os.listdir(data_path)):
        path = os.path.join(data_path, category)
        # print(category, os.listdir(path))
        for movie in tqdm(os.listdir(path)):
            movie_path = os.path.join(path, movie, 'video.mp4')
            cap = cv2.VideoCapture(movie_path)
            frame_id = 0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f'\nProcessing {movie}')
            while cap.isOpened():
                if frame_id % 100 == 0:
                    print(f'frame_id {frame_id}/{frame_count}')
                ret, frame = cap.read()
                if not ret:
                    break
                # cv2.imshow('frame', frame)

                image_data = processor_image(Image.fromarray(frame))
                _, embedding = model_image.encode(image_data, return_features=True)
                embs.append(embedding.flatten())
                df.loc[len(df.index)] = [frame_id, movie, category]
                frame_id += 1

            cap.release()
            cv2.destroyAllWindows()

    embs = np.array(embs)
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, 'database.index')
    df.to_csv('database.csv', index=None)

create_database()