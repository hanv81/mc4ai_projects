import re
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

def main():
    df = pd.read_csv('score.csv', index_col=False)
    clean_data(df)
    create_groups(df)
    
    st.header('BẢNG ĐIỂM LỚP PYTHON AI 09/2022')

    tab1, tab2, tab3, tab4 = st.tabs(['Danh sách', 'Biểu đồ', 'Phân nhóm', 'Phân loại'])
    with tab1:
        df1 = filter1(df)
        df1 = filter2(df1)
        st.write('Số HS:', len(df1), f"({len(df1[df1['GENDER']=='M'])} nam, {len(df1[df1['GENDER']=='F'])} nữ)")
        st.write('GPA: cao nhất ', df1['GPA'].max(), ', thấp nhất ', df1['GPA'].min(), ', trung bình ', df1['GPA'].median())
        st.dataframe(df1, hide_index=True)
    with tab2:
        show_chart(df)
    with tab3:
        clustering(df)
    with tab4:
        classify(df)

def classify(df):
    features = st.multiselect('Chọn đặc trưng', ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S-AVG','BONUS'], ['S6', 'S10', 'S-AVG'])
    if len(features) > 1:
        X = df[features].values.copy()
        y = (df['GPA'] >= 6).values.astype(int)
        model = LogisticRegression()
        model.fit(X,y)
        color = np.where(y == 0, 'red', 'blue')
        text = np.where(y == 0, 'Rớt', 'Đậu')
        b = model.intercept_[0]
        if len(features) == 2:
            w0, w1 = model.coef_[0]
            x0 = np.array([X[:,0].min(), X[:,0].max()])
            x1 = -(x0*w0 + b)/w1
            fig = go.Figure(data=[go.Scatter(x=X[:,0], y=X[:,1], mode='markers', text=df['NAME'], marker=dict(color=color, size=df['GPA'])),
                                  go.Scatter(x=x0, y=x1, mode='lines')],
                            layout=go.Layout({"showlegend": False, "xaxis": {"title":f"{features[0]}"}, "yaxis": {"title":f"{features[1]}"}}))
            st.plotly_chart(fig)
        elif len(features) == 3:
            w0, w1, w2 = model.coef_[0]
            x1 = np.array([X[:,0].min(), X[:,0].max()])
            y1 = np.array([X[:,1].min(), X[:,1].max()])
            xx, yy = np.meshgrid(x1, y1)
            xy = np.c_[xx.ravel(), yy.ravel()]
            z = -(w0*xy[:,0] + w1*xy[:,1] + b)/w2
            fig = go.Figure(data=[go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers', text=df['NAME'], 
                                               marker=dict(color=color, size=df['GPA'])),
                                  go.Surface(x=x1, y=y1, z=z.reshape(xx.shape))])
            fig.update_layout(scene=dict(xaxis_title=f"{features[0]}", yaxis_title=f"{features[1]}", zaxis_title=f"{features[2]}"))
            st.plotly_chart(fig)

        st.write('Độ chính xác:', model.score(X, y).round(2))

def clustering(df):
    df['S-AVG'] = (df['S1'] + df['S2'] + df['S3'] + df['S4'] + df['S5'] + df['S7'] + df['S8'] + df['S9'])/8
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider('Số nhóm', value=3, min_value=2, max_value=5)
    with col2:
        features = st.multiselect('Chọn đặc trưng', ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S-AVG','BONUS'], 
                                  ['S6', 'S10', 'S-AVG'], key=0)

    if len(features) > 0:
        X = df[features].values.copy()
        text = np.where(df['GPA'] >= 6, 'Đậu', 'Rớt')
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
        kmeans.fit(X)
        y = kmeans.labels_
        centers = kmeans.cluster_centers_
        if len(features) in (2,3):
            df_cluster = df[features + ['GPA', 'NAME']].copy()
            df_cluster['Nhóm'] = (y+1).astype(str)
            if len(features) == 2:
                fig = px.scatter(df_cluster, x=features[0], y=features[1], color='Nhóm', size='GPA', hover_name='NAME')
            else:
                fig = px.scatter_3d(df_cluster, x=features[0], y=features[1], z=features[2], color='Nhóm', size='GPA', hover_name='NAME')
            fig.layout.update(showlegend=False)
            st.plotly_chart(fig)

        for i in range(n_clusters):
            col1,col2 = st.columns(2)
            with col1:
                df_cluster = df[y == i][['NAME'] + features + ['GPA']]
                st.markdown(f'''**Nhóm {i+1}**
- GPA cao nhất :red[{df_cluster['GPA'].max()}]
- GPA Thấp nhất :green[{df_cluster['GPA'].min()}]
- GPA Trung bình :blue[{round(df_cluster['GPA'].mean(),2)}]''')
                st.dataframe(df_cluster, hide_index=True)
            with col2:
                color = np.where(df['GPA'][y==i] < 6, 'red', 'blue')
                if len(features) == 2:
                    data = [go.Scatter(x=X[y==i,0], y=X[y==i,1], mode='markers', text=df['NAME'][y==i], marker=dict(size=df['GPA'][y==i], color=color)),
                            go.Scatter(x=centers[i:i+1,0], y=centers[i:i+1,1], mode='markers', marker=dict(symbol='star-diamond', size=8), marker_color='rgb(125, 255, 0)')]

                    fig = go.Figure(data=data, layout=go.Layout({'showlegend':False, "xaxis": {"title":f"{features[0]}"}, "yaxis": {"title":f"{features[1]}"}}))
                    st.plotly_chart(fig)
                elif len(features) == 3:
                    data = [go.Scatter3d(x=X[y==i,0], y=X[y==i,1], z=X[y==i,2], mode='markers', name=f'Nhóm {i+1}', text=df['NAME'][y==i], marker=dict(size=df['GPA'][y==i], color=color)),
                            go.Scatter3d(x=centers[i:i+1,0], y=centers[i:i+1,1], z=centers[i:i+1,2], mode='markers', marker=dict(symbol='diamond', size=4), marker_color='rgb(125, 255, 0)')]
                    fig = go.Figure(data=data)
                    fig.update_layout(showlegend=False, scene=dict(xaxis_title=f"{features[0]}", yaxis_title=f"{features[1]}", zaxis_title=f"{features[2]}"))
                    st.plotly_chart(fig)

def filter1(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write('Giới tính')
        male = st.checkbox('Nam', value=True)
        female = st.checkbox('Nữ', value=True)
        if not male:
            df = df[df['GENDER'] != 'M']
        if not female:
            df = df[df['GENDER'] != 'F']
    with col2:
        st.write('Khối lớp')
        cl = [st.checkbox(f'Lớp {i}', value=True, key=i) for i in (10,11,12)]
        for i in range(3):
            if not cl[i]:
                df = df[~df['CLASS'].str.startswith(f'{i+10}')]
    with col3:
        room = st.selectbox('Phòng', options=['Tất cả', 'A114', 'A115'])
        if room != 'Tất cả':
            df = df[df['PYTHON-CLASS'].str.startswith(room[1:])]
    with col4:
        time = st.multiselect('Buổi', options=['Sáng', 'Chiều'])
        if len(time) == 1:
            df = df[df['PYTHON-CLASS'].str.endswith(time[0][0])]
    return df

def filter2(df):
    st.write('Lớp chuyên')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        cvan = st.checkbox('Văn', value=True)
        if not cvan:
            df = df[~df['CLASS'].str.match('^..CV.$')]
        math = st.checkbox('Toán', value=True)
        if not math:
            df = df[~df['CLASS'].str.match('^..CT.$')]
    with col2:
        physics = st.checkbox('Lý', value=True)
        if not physics:
            df = df[~df['CLASS'].str.match('^..CL.$')]
        chemist = st.checkbox('Hoá', value=True)
        if not chemist:
            df = df[~df['CLASS'].str.match('^..CH.$')]
    with col3:
        en = st.checkbox('Anh', value=True)
        if not en:
            df = df[~df['CLASS'].str.match('^..CA.$')]
        ctin = st.checkbox('Tin', value=True)
        if not ctin:
            df = df[~df['CLASS'].str.contains('CTIN')]
    with col4:
        csd = st.checkbox('Sử Địa', value=True)
        if not csd:
            df = df[~df['CLASS'].str.contains('CSD')]
        csd = st.checkbox('Trung Nhật', value=True)
        if not csd:
            df = df[~df['CLASS'].str.contains('CTRN')]
    with col5:
        thsn = st.checkbox('TH/SN', value=True)
        if not thsn:
            df = df[~(df['CLASS'].str.contains('TH') | df['CLASS'].str.contains('SN'))]
        others = st.checkbox('Khác', value=True)
        if not others:
            df = df[~(df['CLASS'].str.match('^..[A-B]'))]

    return df

def clean_data(df):
    df.fillna({'BONUS':0, 'REG-MC4AI':'N'}, inplace=True)
    df.fillna({f'S{i}':0 for i in range(1,11)}, inplace=True)

def create_groups(df):
    def group_filter(row):
        if re.search('CV', row['CLASS']):
            return 'Chuyên Văn'
        if re.search('CTIN', row['CLASS']):
            return 'Chuyên Tin'
        if re.search('CTRN', row['CLASS']):
            return 'Trung Nhật'
        if re.search('CT', row['CLASS']):
            return 'Chuyên Toán'
        if re.search('CL', row['CLASS']):
            return 'Chuyên Lý'
        if re.search('CH', row['CLASS']):
            return 'Chuyên Hoá'
        if re.search('CA', row['CLASS']):
            return 'Chuyên Anh'
        if re.search('CSD', row['CLASS']):
            return 'Sử Địa'
        if re.search('[TH|SN]', row['CLASS']):
            return 'TH/SN'
        return 'Khác'
    df['CLASS-GROUP'] = df.apply(group_filter, axis=1)

def show_chart(df):
    tab1, tab2 = st.tabs(['Số lượng HS', 'Điểm'])
    with tab1:
        fig = px.pie(df, names='PYTHON-CLASS')
        st.plotly_chart(fig)
        st.success('Kết luận: Số học sinh ở 2 buổi gần bằng nhau, nên giờ học là hợp lý, đáp ứng được nhu cầu của tất cả học sinh')

        fig = px.pie(df, names='CLASS-GROUP')
        st.plotly_chart(fig)
        st.success('''Kết luận:
- Khối Chuyên Toán quan tâm đến AI nhiều nhất
- ...''')

        fig = px.pie(df, names='GENDER')
        st.plotly_chart(fig)
        st.success('Kết luận: Nhìn chung học sinh Nam hứng thú với AI hơn học sinh Nữ')

    with tab2:
        s = st.radio('Điểm từng Session', options=['S1','S2','S3','S4','S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'GPA'], horizontal=True)
        fig = px.box(df, x='PYTHON-CLASS',  y=s, color='GENDER')
        st.plotly_chart(fig)
        st.info('Kết luận: Nhìn chung học sinh Nam học tốt hơn học sinh Nữ và lớp 114 học tốt hơn lớp 115')

        fig = px.box(df, x='CLASS-GROUP',  y=s)
        st.plotly_chart(fig)
        st.info('''Kết luận:
- Khối Chuyên Tin học tốt nhất
- Khối Trung Nhật/Tích Hợp/Song Ngữ học kém nhất
- Bất ngờ nhất là khối Chuyên Văn, đậu 100% ...''')

        df1 = df[[s, 'CLASS-GROUP', 'GENDER']].groupby(['CLASS-GROUP','GENDER']).median()
        df1.reset_index(inplace=True)
        fig = px.bar(df1, x='CLASS-GROUP',  y=s, color='GENDER', barmode='group')
        st.plotly_chart(fig)

        df1 = df[s].value_counts()
        st.plotly_chart(px.bar(df1, x=df1.index, y=df1.values))

main()