import traceback
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def mse(y, y_pred):
  return ((y-y_pred)**2).mean()

def bce(y, y_pred):
  e = .1e-10
  return -np.mean(y*np.log(e+y_pred) + (1-y)*np.log(e+1-y_pred))

def bce_loss(y, y_pred):
  e = .1e-10
  yy = np.tile(y, (y_pred.shape[0],1))
  return (-yy*np.log(e+y_pred) - (1-yy)*np.log(e+1-y_pred)).mean(axis=1)

def mse_loss(y, y_pred):
  yy = np.tile(y, (y_pred.shape[0],1))
  return ((y_pred - yy)**2).mean(axis=1)

def accuracy(y, y_pred):
  return (np.abs(y-y_pred) <= .5).sum()/y.shape[0]

def feed_forward(X, w):
  return 1/(1 + np.exp(-(X@w)))

def gradient(X, y, y_pred, loss_fn):
  if loss_fn == 'BCE':
    return (X*(y_pred-y).reshape(-1,1)).mean(axis=0)
  return (X*(((y_pred-y)*y_pred*(1-y_pred)).reshape(-1,1))).mean(axis=0)

def back_propagation(w, dw, lr):
  return w-lr*dw

def generate_weights(n, start_point):
  try:
    start_point = start_point.split()
    if len(start_point) == 2: return np.array(start_point, dtype=float)
  except:
    st.info('Random init')
  return np.random.rand(n)

def draw_result(X, y, history):
  w,b = history['weights'][np.argmin(history['loss'])]
  st.write('Optimal weights:', w,b)
  xx = np.linspace(X.min()-.5, X.max()+.5)
  yy = 1/(1 + np.exp(-xx*w - b))
  color = np.where(y == 0, 'rgba(255, 0, 0, .8)', 'rgba(0, 0, 255, .8)')

  fig = make_subplots(rows=1, cols=2, subplot_titles=('History', 'Result'))
  fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Loss'), row=1, col=1)
  fig.add_trace(go.Scatter(y=history['accuracy'], mode='lines', name='Accuracy'), row=1, col=1)
  fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', marker=dict(color=color)), row=1, col=2)
  fig.add_trace(go.Scatter(x=xx, y=yy, mode='lines', name='Class 1 Probability', marker_color='rgba(0, 0, 255, .8)'), row=1, col=2)
  fig.add_trace(go.Scatter(x=xx, y=1-yy, mode='lines', name='Class 2 Probability', marker_color='rgba(255, 0, 0, .8)'), row=1, col=2)
  fig.add_trace(go.Scatter(x=[-b/w], y=[.5], name='Decision Point', marker_color='rgba(0, 0, 0, .8)'), row=1, col=2)
  fig.update_layout(showlegend=False)
  st.plotly_chart(fig)

def visualize_gd(X, y, history, loss_fn, range, num):
  w1,w2,b1,b2 = range
  w = np.linspace(w1, w2, num)
  b = np.linspace(b1, b2, num)
  ww, bb = np.meshgrid(w, b)
  wb = np.c_[ww.ravel(), bb.ravel()]
  w_, b_ = wb[:,0], wb[:,1]
  y_pred = (1/(1 + np.exp(-(w_*X + b_)))).T
  loss_face = bce_loss(y, y_pred) if loss_fn == 'BCE' else mse_loss(y, y_pred)

  weights = np.array(history['weights'])
  id = np.argmin(history['loss'])
  fig = go.Figure(data=[
      go.Surface(x=w, y=b, z=loss_face.reshape(ww.shape), name='Loss Surface'),
      go.Scatter3d(x=weights[:,0], y=weights[:,1], z=history['loss'], mode='markers', name='Learning Route'),
      go.Scatter3d(x=[weights[0,0]], y=[weights[0,1]], z=[history['loss'][0]], mode='markers', name='Initial'),
      go.Scatter3d(x=[weights[id,0]], y=[weights[id,1]], z=[history['loss'][id]], mode='markers', name='Optimal'),
  ])

  fig.update_layout(scene=dict(xaxis_title='w', yaxis_title='b', zaxis_title='L'))
  st.plotly_chart(fig)

def train(X, y, lr, epochs, loss_fn, start_point):
  w = generate_weights(X.shape[1]+1, start_point)
  X_ = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
  history = {'loss':[], 'accuracy':[], 'weights':[]}
  for _ in range(epochs):
    y_pred = feed_forward(X_, w)
    loss = bce(y, y_pred) if loss_fn == 'BCE' else mse(y, y_pred)
    acc = accuracy(y, y_pred)
    history['weights'].append(w)
    history['loss'].append(loss)
    history['accuracy'].append(acc)
    dw = gradient(X_, y, y_pred, loss_fn)
    w = back_propagation(w, dw, lr)

  w = history['weights'][np.argmin(history['loss'])]
  y_pred = feed_forward(X_, w)
  loss = bce(y, y_pred)
  acc = accuracy(y, y_pred)
  return history

def create_dataset():
  with st.expander('Data'):
    col1,col2 = st.columns(2)
    with col1:
      X = st.text_area('X', '0.50 0.75 1.00 1.25 1.50 1.75 1.75 2.00 2.25 2.50 2.75 3.00 3.25 3.50 4.00 4.25 4.50 4.75 5.00 5.50')
    with col2:
      y = st.text_area('y', '0 0 0 0 0 0 1 0 1 0 1 0 1 0 1 1 1 1 1 1')
    try:
      X = np.array(X.split(), dtype=float).reshape(-1,1)
      y = np.array(y.split(), dtype=int)
      if X.shape[0] != y.shape[0]:
        st.error('len(X) != len(y)')
      else:
        cols = st.columns(5)
        with cols[0]:draw_bce = st.toggle('Draw BCE')
        with cols[1]:draw_mse = st.toggle('Draw MSE')
        if draw_bce + draw_mse == 0:
          color = np.where(y == 0, 'rgba(255, 0, 0, .8)', 'rgba(0, 0, 255, .8)')
          fig = go.Figure(data=go.Scatter(x=X.flatten(), y=y, mode='markers', marker=dict(color=color)))
        else:
          col1, col2, col3 = st.columns(3)
          with col1:
            w1 = st.number_input('w1', value=-50)
            w2 = st.number_input('w2', value=50)
          with col2:
            b1 = st.number_input('b1', value=-50)
            b2 = st.number_input('b2', value=50)
          with col3:num = st.number_input('Points to linspace', value= 100, min_value=50, max_value=500, step=50)
          w = np.linspace(w1, w2, num)
          b = np.linspace(b1, b2, num)
          ww, bb = np.meshgrid(w, b)
          wb = np.c_[ww.ravel(), bb.ravel()]
          w_, b_ = wb[:,0], wb[:,1]
          y_pred = (1/(1 + np.exp(-(w_*X + b_)))).T
          data = []
          if draw_bce:
            data.append(go.Surface(x=w, y=b, z=bce_loss(y, y_pred).reshape(ww.shape), name='BCE'))
          if draw_mse:
            data.append(go.Surface(x=w, y=b, z=mse_loss(y, y_pred).reshape(ww.shape), name='MSE'))
          fig = go.Figure(data=data)
          fig.update_layout(scene=dict(xaxis_title='w', yaxis_title='b', zaxis_title='L'))
        st.plotly_chart(fig)
    except:
      traceback.print_exc()
      st.error('Error data input')
  return X,y

def main():
  X,y = create_dataset()

  with st.expander('Training Info', True):
    cols = st.columns(4)
    with cols[0]:
      lr = st.selectbox('Learning Rate', (.1, .05, .01, .005, .001, .0005, .0001), 1)
    with cols[1]:
      epochs = st.number_input('Epochs', value=2000, min_value=1000, step=200)
    with cols[2]:
      loss_fn = st.radio('Loss', ('BCE', 'MSE'))
    with cols[3]:
      start_point = st.text_input('Start point', '10 30')

  with st.expander('Visualize Range'):
    col1, col2, col3 = st.columns(3)
    with col1:
      w1 = st.number_input('w1', value=-50, key='w1')
      w2 = st.number_input('w2', value=50, key='w2')
    with col2:
      b1 = st.number_input('b1', value=-50, key='b1')
      b2 = st.number_input('b2', value=50, key='b2')
    with col3:num = st.number_input('Points to linspace', value= 100, min_value=50, max_value=500, step=50, key='num')

  if st.button('Train', use_container_width=True):
    history = train(X, y, lr, epochs, loss_fn, start_point)
    visualize_gd(X, y, history, loss_fn, [w1,w2,b1,b2], num)
    draw_result(X,y,history)

main()