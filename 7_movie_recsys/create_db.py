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
    df = pd.DataFrame(columns = ['Frame ID', 'Path'])
    
    for category in tqdm(os.listdir(data_path)):
        path = os.path.join(data_path, category)
        # print(category, os.listdir(path))
        for movie in tqdm(os.listdir(path)):
            movie_path = os.path.join(path, movie)
            for filename in os.listdir(movie_path):
                if not filename.endswith('mp4'):
                    continue
                
                file_path = os.path.join(movie_path, filename)
                cap = cv2.VideoCapture(file_path)
                frame_id = 0
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f'\nProcessing {filename}/{frame_count}', )
                while cap.isOpened():
                    # if frame_id % 100 == 0:
                    #     print(f'frame_id {frame_id}/{frame_count}')
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # cv2.imshow('frame', frame)

                    image_data = processor_image(Image.fromarray(frame))
                    _, embedding = model_image.encode(image_data, return_features=True)
                    embs.append(embedding.flatten())
                    frame_id += 1
                    df.loc[len(df.index)] = [frame_id, file_path]

                cap.release()
                cv2.destroyAllWindows()

    embs = np.array(embs)
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, 'database.index')
    df.to_csv('database.csv')

create_database()