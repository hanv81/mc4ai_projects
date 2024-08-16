import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from stqdm import stqdm
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

@st.cache_data
def create_dataset(n_samples):
  x_1 = np.random.rand(n_samples)
  x_2 = np.random.rand(n_samples)
  y = np.array([1 if (i >= 0.4 and j >= 0.4) or i + j >= 1.1 else 0 for i,j in zip(x_1, x_2)])
  X = np.concatenate((x_1.reshape(-1,1), x_2.reshape(-1,1)), axis=1)
  return X, y

e = .1e-10
def bce_loss(y, y_pred):
  return -np.mean(y*np.log(e+y_pred) + (1-y)*np.log(e+1-y_pred))

def accuracy(y, y_pred, threshold=.5):
  y_hat = [0 if i < threshold else 1 for i in y_pred]
  return (y==y_hat).sum()/y.shape[0]

def feed_forward(X, w):
  return 1/(1 + np.exp(-(X@w)))

def gradient(X, y, y_pred):
  return (X*(y_pred-y).reshape(-1,1)).sum(axis=0)

def gradient_descent(x, y, w, eta):
  y_pred = feed_forward(x, w)
  w = w - eta * gradient(x, y, y_pred)
  return w, y_pred

def generate_weights(n_features):
  return np.random.rand(n_features+1)

@st.cache_data
def fit(X, y, ETA, EPOCHS, batch_size=0):
  w = generate_weights(X.shape[1])
  history = {'loss':[], 'accuracy':[], 'weights':[]}
  X_ = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
  for i in stqdm(range(EPOCHS)):
    if batch_size > 0:
      index = np.random.permutation(X.shape[0])
      for i in range(X.shape[0]//batch_size):
        j = i * batch_size
        id = index[j:j+batch_size]
        w,_ = gradient_descent(X_[id], y[id], w, ETA)
      y_pred = feed_forward(X_, w)
    else:
      w, y_pred = gradient_descent(X_, y, w, ETA)
    loss = bce_loss(y, y_pred)
    acc = accuracy(y, y_pred)
    history['weights'].append(w)
    history['loss'].append(loss)
    history['accuracy'].append(acc)

  return history

def visualize_decision_boundary(X, y, history, history_batch, threshold):
  w = np.array(history_batch['weights']) if history_batch else np.array(history['weights'])
  x1 = np.array([X[:, 0].min()-.05, X[:, 0].max()+.05])
  x2 = -(x1*w[-1][0] + w[-1][2])/w[-1][1]
  data = [go.Scatter(x=x1, y=x2, mode='lines', name = 'Threshold 0.5', marker=dict(color='yellowgreen')),
          go.Scatter(x=X[y==0,0], y=X[y==0,1], mode='markers', name='Class 0', marker=dict(color='orange')),
          go.Scatter(x=X[y==1,0], y=X[y==1,1], mode='markers', name='Class 1', marker=dict(color='blue'))
          ]
  if threshold != .5:
    x2_t = -(x1*w[-1][0] + w[-1][2] + np.log(1/threshold-1))/w[-1][1]
    data.append(go.Scatter(x=x1, y=x2_t, mode='lines', name = f'Threshold {threshold}', line=dict(color='red', dash='dash')))
  layout = go.Layout(showlegend=True, title="Decision Boundary",
                     xaxis=dict(range=[X[:,0].min()-.05, X[:,0].max()+.05], autorange=False),
                     yaxis=dict(range=[X[:,1].min()-.05, X[:,1].max()+.05], autorange=False),
                     updatemenus=[dict(type="buttons", buttons=[dict(label="Play",method="animate",args=[None])])]
                     )
  frames=[go.Frame(data=[go.Scatter(x=x1, y=(-x1*w[i][0] - w[i][2])/w[i][1], mode='lines', name = 'Threshold 0.5', line=dict(color='yellowgreen'))])
                         for i in range(len(w))]
  fig = go.Figure(data=data, layout=layout, frames=frames)
  st.plotly_chart(fig)

def visualize_history(history, history_batch):
  w = np.array(history_batch['weights']) if history_batch else np.array(history['weights'])
  fig = make_subplots(rows=1, cols=2, subplot_titles=('History', 'Learning Route'),
                      specs=[[{'type':'xy'}, {'type':'surface'}]])
  if history_batch:
    fig.add_trace(go.Scatter(y=history_batch['loss'], mode='lines', name='Mini-batch Loss'), row=1, col=1)
  fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Batch Loss', line = dict(color='magenta')), row=1, col=1)
  fig.add_trace(go.Scatter(y=history['accuracy'], mode='lines', name='Accuracy'), row=1, col=1)

  loss = np.array(history_batch['loss']) if history_batch else np.array(history['loss'])
  fig.add_trace(go.Scatter3d(x=w[:,0], y=w[:,1], z=w[:,2], mode='markers', name='Loss',
                             text=loss, marker=dict(size=loss*20, color='red')), row=1, col=2)
  fig.update_xaxes(title_text="Epochs", row=1, col=1)
  st.plotly_chart(fig)

def show_report(X, y, history, threshold):
  col1, col2 = st.columns(2)
  with col1:
    w = history['weights'][-1]
    X_ = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    y_pred = feed_forward(X_, w)
    y_pred_label = [0 if i < threshold else 1 for i in y_pred]
    cm = confusion_matrix(y, y_pred_label)
    fig = plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    st.pyplot(fig)
  with col2:
    st.subheader('Classification Report')
    st.text(classification_report(y, y_pred_label))

def train(X, y, eta, epochs, batch_size=0):
  t = time.time()
  history = fit(X, y, eta, epochs, batch_size)
  t = (time.time() - t)*1000
  return history, int(t)

def show_result(X, y, history, history_batch, threshold):
  with st.spinner('Visualizing...'):
    visualize_decision_boundary(X, y, history, history_batch, threshold)
    visualize_history(history, history_batch)
  show_report(X, y, history, threshold)

def main():
  st.header('Logistic Regression')
  col1, col2, col3, col4 = st.columns(4)
  with col1:
    n_samples = st.number_input('Number of Samples', value=200, min_value=100, max_value=10000, step=100)
  with col2:
    eta = st.number_input('Learning Rate', max_value=.1, value=.01)
  with col3:
    epochs = st.number_input('Epochs', value=300, step=50, min_value=10)
  with col4:
    batch_train = st.toggle('Mini-Batch GD')
    batch_size = st.number_input('Batch Size', min_value=1, max_value=100, value=20, step=5)
    if not batch_train:batch_size = 0
  threshold = st.slider('Threshold', min_value=.01, max_value=.99, value=.5, step=.01)

  X,y = create_dataset(n_samples)
  history, t = train(X, y, eta, epochs)
  with st.expander('Training Info'):
    st.write('Batch training time:', t, 'ms. Accuracy:', round(history['accuracy'][-1]*100,2), 'Loss:', round(history['loss'][-1],4))
    history_batch = None
    if batch_train:
      history_batch, t_batch = train(X, y, eta, epochs, batch_size)
      w = history_batch['weights'][-1]
      X_ = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
      y_pred = feed_forward(X_, w)
      loss = round(bce_loss(y, y_pred),4)
      acc = round(accuracy(y, y_pred)*100,2)
      st.write('Mini-batch training time:', t_batch, 'ms. Accuracy:', acc, 'Loss:', loss)

  show_result(X, y, history, history_batch, threshold)

if __name__ == "__main__":
  main()