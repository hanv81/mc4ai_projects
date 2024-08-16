import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
from streamlit_drawable_canvas import st_canvas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Input

DS_PATH = 'dataset'

@st.cache_data
def read_data(n = 1000):
    labels = os.listdir(DS_PATH)
    X = None
    y = None
    for i in tqdm(range(len(labels))):
        subfolder = os.listdir(os.path.join(DS_PATH, labels[i]))
        imgs = [Image.open(os.path.join(DS_PATH, labels[i], file)) for file in subfolder[:n]]
        imgs = np.stack([np.array(img, dtype=float) for img in imgs])
        # print(labels[i], imgs.shape)
        X = imgs if X is None else np.concatenate((X, imgs))
        if y is None:y = [i] * len(imgs)
        else:y.extend([i] * len(imgs))
    
    y = np.array(y)
    # print(X.shape, y.shape)
    return X,y,np.array(labels)

def preprocess(X, y, test_size):
    X_train = X / 255
    num_classes = max(y) + 1
    X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=test_size, stratify=y)
    y_train_ohe = to_categorical(y_train, num_classes=num_classes)
    y_test_ohe = to_categorical(y_test, num_classes=num_classes)
    return X_train, X_test, y_train_ohe, y_test_ohe

def train(X, y, epochs, num_classes):
    model = Sequential()
    model.add(Input(shape=X.shape[1:]))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    history = model.fit(X, y, epochs = epochs, verbose=1)
    model.save('model.h5')
    return model, history

def read_batch_data(zip, files, i, batch_size, data):
    for j in range(i, min(i+batch_size,len(files))):
        if files[j].endswith('png'):
            with zip.open(files[j]) as f:
                data[files[j]] = np.array(Image.open(f), dtype=float)

def read_zip_file(uploaded_file, labels):
    data = {}
    batch_size = 32
    with ZipFile(uploaded_file, 'r') as zip:
        with ThreadPoolExecutor() as executor:
            files = zip.namelist()
            for i in range(0, len(files), batch_size):
                executor.submit(read_batch_data, zip, files, i, batch_size, data)
    
    X = np.array(list(data.values()))
    y = [k.split('/')[1] for k in data.keys()]
    lb_list = labels.tolist()
    y = np.array([lb_list.index(i) for i in y])
    # print(X.shape, y.shape)
    return X, y

def training():
    n = st.slider('Number of samples per class', min_value=1000, max_value=20000, step=200)
    X, y, labels = read_data(n)
    epochs = st.slider('Epochs', min_value=5, max_value=50, value=10, step=5)
    test_size = st.slider('Test size', min_value=.05, max_value=.5, value=0.1, step=.05)
    uploaded_file = st.file_uploader('Upload Dataset', type=['zip'])
    
    if st.button('Train'):
        if uploaded_file is not None:
            with st.spinner('Reading zip file...'):
                X, y = read_zip_file(uploaded_file, labels)

        with st.spinner('Training...'):
            X_train, X_test, y_train_ohe, y_test_ohe = preprocess(X, y, test_size)
            model,history = train(X_train, y_train_ohe, epochs, y.max()+1)
            _, accuracy = model.evaluate(X_test, y_test_ohe)
            st.success(f'Done. Accuracy on test set: {round(accuracy,2)}')
            fig, _ = plt.subplots(1,2)
            fig.set_figheight(2)
            plt.subplot(1,2,1)
            plt.title('Loss')
            plt.plot(history.history['loss'])
            plt.subplot(1,2,2)
            plt.title('Accuracy')
            plt.plot(history.history['accuracy'])
            st.pyplot(fig)

    return X, y, labels

def inference(X, y, labels):
    st.subheader('Draw a letter (A-Z) or upload an image')
    col1, col2 = st.columns(2)
    with col1:
        canvas_result = st_canvas(
            stroke_width=15,
            stroke_color='rgb(255, 255, 255)',
            background_color='rgb(0, 0, 0)',
            height=150,
            width=150,
            key="canvas",
        )
    if canvas_result.image_data is not None:img = Image.fromarray(canvas_result.image_data)

    with col2:
        uploaded_file = st.file_uploader('', type=['png','jpg','bmp'])
        if uploaded_file is not None:img = Image.open(uploaded_file)

    if st.button('Predict'):
        model = load_model('model.h5')
        input_shape = X[0].shape
        img = img.resize(input_shape).convert('L')
        img = np.array(img, dtype=float)/255
        if uploaded_file is not None: 
            col3, col4, col5 = st.columns(3)
            with col3:
                st.text('Original Image')
                st.image(Image.open(uploaded_file))
            with col4:
                st.text('Grayscale Image')
                inference_image(model, labels, img, input_shape)
            with col5:
                st.text('Invert Grayscale Image')
                inference_image(model, labels, 1-img, input_shape)
        else:
            col6, col7 = st.columns(2)
            with col6:
                st.caption('Softmax')
                inference_image(model, labels, img, input_shape, False)
            with col7:
                st.caption('KNN')
                inference_knn(X, y, labels, img, input_shape)

def inference_image(model, labels, img, input_shape, draw_img=True):
    probs = model.predict(img.reshape(-1,*input_shape)).squeeze()*100
    ids = np.argsort(probs)[::-1]
    if draw_img:st.image(img)
    for i in ids[:5]:
        st.write(labels[i], ':', probs[i].round(decimals=2), '%')

@st.cache_data
def create_knn_model(X, y, input_shape):
    return KNeighborsClassifier().fit(X.reshape(-1, input_shape[0]*input_shape[1]), y)

def inference_knn(X, y, labels, img, input_shape):
    knn = create_knn_model(X, y, input_shape)
    id = knn.predict([img.flatten()]).squeeze()
    st.write(labels[id])

def main():
    st.title('HANDWRITING RECOGNITION')
    with st.sidebar:
        X, y, labels = training()
    inference(X, y, labels)

main()