import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from stqdm import stqdm

@st.cache_data
def create_dataset(n = 300):
  means = [0, 0]
  cov = [[1, 0], [0, 1]]
  return np.random.multivariate_normal(means, cov, n)

e = .1e-10

def ce_loss(y, y_pred):
  # plus e for prevent zero log
  ce = np.array([-(y[i]*np.log(y_pred[i] + e)).sum() for i in range(len(y))])
  return ce.mean()

def accuracy(y, y_pred):
  return (y == y_pred).sum()/y.shape[0]

def one_hot_encoding(y):
  c = int(max(y)) + 1 # num classes
  return np.array([[1 if j==y[i] else 0 for j in range(c)] for i in range(len(y))])

def feed_forward(X, W):
  y = np.exp(X @ W.T)
  return y/np.tile(y.sum(axis=1), (y.shape[1],1)).T

def gradient(X, y, y_pred):
  return (y_pred-y).T @ X

def generate_weights(n_classes, n_features):
  return np.random.rand(n_classes, n_features+1)

@st.cache_data
def kmeans(X, n_clusters):
  t = time.time()
  history = []
  centers = X[np.random.choice(X.shape[0], n_clusters, replace = False)]
  while True:
    d = [[np.linalg.norm(x-c) for c in centers] for x in X]
    y = np.argmin(d, axis=1)

    centers_new = np.array([np.mean(X[y==i], axis=0) for i in range(n_clusters)])

    history.append((centers, y))
    if np.array_equal(centers, centers_new):
      t = (time.time() - t)*1000
      return history, t
    centers = centers_new

@st.cache_data
def train(X, y, ETA, EPOCHS, batch_size=0):
  t = time.time()
  
  y_ohe = one_hot_encoding(y)
  W = generate_weights(int(max(y))+1, X.shape[1])
  history = {'loss':[], 'accuracy':[], 'weights':[]}
  X_ = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
  for i in stqdm(range(EPOCHS)):
    if batch_size > 0:
      id = np.random.choice(len(y), batch_size)
      XX, yy, yy_ohe = X_[id], y[id], y_ohe[id]
      if batch_size == 1:
        XX = XX.reshape(1,-1)
        yy = yy.reshape(1,-1)
        yy_ohe = yy_ohe.reshape(1,-1)
    else:
      XX, yy, yy_ohe = X_, y, y_ohe
    y_pred = feed_forward(XX, W)
    if batch_size == 1:y_pred = y_pred.reshape(1,-1)
    loss = ce_loss(yy_ohe, y_pred)
    acc = accuracy(yy, y_pred.argmax(axis=1))
    history['loss'].append(loss)
    history['accuracy'].append(acc)
    history['weights'].append(W)
    # if i%10==0:
    #   print(f'iter {i}, loss: {loss}, accuracy: {acc}')

    dW = gradient(XX, yy_ohe, y_pred)
    W = W - ETA*dW

  W = history['weights'][np.argmin(history['loss'])]
  y_pred = feed_forward(X_, W)
  loss = ce_loss(y_ohe, y_pred)
  acc = accuracy(y, y_pred.argmax(axis=1))
  t = (time.time() - t)*1000
  return history, loss, acc, t

def visualize_clustering_history(X, history, n_clusters):
  for i in range(len(history)):
    centers, y = history[i]
    X_ = np.concatenate((X, centers))
    y_ = np.concatenate((y, [n_clusters]*n_clusters))
    history[i] = X_, y_

  X_, y_  = history[-1]
  symbol = np.where(y_ < n_clusters, 'circle', 'star')
  size = np.where(y_ < n_clusters, 7, 12)
  data = [go.Scatter(x=history[0][0][:,0], y=history[0][0][:,1], mode='markers', marker=dict(color=history[0][1], symbol=symbol, size=size))]
  frames=[go.Frame(data=[go.Scatter(x=history[i][0][:,0], y=history[i][0][:,1], mode='markers', text=history[i][1],
                                    marker=dict(color=history[i][1], symbol=symbol, size=size))])
          for i in range(len(history))]
  layout=go.Layout(title='Clustering', showlegend=False, hovermode="closest", xaxis_title='x1', yaxis_title='x2',
                   updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate",
                                     args=[None, {"frame": {"duration": 1000, "redraw": False},}])])])
  fig = go.Figure(data=data, layout=layout, frames=frames)
  st.plotly_chart(fig)

def visualize_softmax_history(history):
  fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Accuracy'))
  fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Loss', line = dict(color='magenta')), row=1, col=1)
  fig.add_trace(go.Scatter(y=history['accuracy'], mode='lines', name='Accuracy'), row=1, col=2)
  fig.update_xaxes(title_text="Epochs", row=1, col=1)
  fig.update_xaxes(title_text="Epochs", row=1, col=2)
  fig.update_layout(showlegend=False, title='History')
  st.plotly_chart(fig)

def main():
  st.header('Clustering & Softmax Regression')
  col1, col2, col3, col4 = st.columns(4)
  with col1:
    n_samples = st.number_input('Number of Samples', value=100, min_value=50, max_value=1000, step=50)
    n_clusters = st.number_input('Number of Clusters', value=3, min_value=2, max_value=10)

  with col2:
    eta = st.number_input('Learning Rate', max_value=.1, value=.01)
  with col3:
    epochs = st.number_input('Epochs', value=300, step=10, min_value=10)
  with col4:
    batch_train = st.toggle('Mini-Batch GD')
    batch_size = st.number_input('Batch Size', min_value=1, max_value=100, value=20, step=5)
    if not batch_train:batch_size = 0

  X = create_dataset(n_samples)
  clustering_history, t = kmeans(X, n_clusters)
  centers, y = clustering_history[-1]
  history, loss, acc, t_train = train(X, y, eta, epochs, batch_size)

  with st.expander('Training Info'):
    st.write('Clustering time:', int(t))
    st.write('Training time:', int(t), 'Loss:', round(loss,4), 'Accuracy:', round(acc*100,2))

  visualize_clustering_history(X, clustering_history, n_clusters)
  visualize_softmax_history(history)

if __name__ == "__main__":
  main()