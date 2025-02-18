import streamlit as st
import requests
import zipfile
import io
import pandas as pd
import os
import gdown
import tempfile

import plotly.graph_objs as go
import streamlit as st

def create_stylish_line_plot(df, x_col, y1_col, y2_col, title="Stylish Line Plot", x_label="X", y_label="Values"):
    """
    Membuat line plot yang menarik dengan dua kolom y berbeda dan kolom x sebagai sumbu x.

    Parameters:
    - df: DataFrame yang berisi data.
    - x_col: Nama kolom yang akan digunakan sebagai sumbu x.
    - y1_col: Nama kolom yang akan digunakan sebagai garis pertama.
    - y2_col: Nama kolom yang akan digunakan sebagai garis kedua.
    - title: Judul plot.
    - x_label: Label untuk sumbu x.
    - y_label: Label untuk sumbu y.
    """
    
    # Membuat trace untuk y1
    trace1 = go.Scatter(
        x=df[x_col],
        y=df[y1_col],
        mode='lines+markers',
        name=f'{y1_col}',
        line=dict(color='dodgerblue', width=2),
        marker=dict(size=8)
    )

    # Membuat trace untuk y2
    trace2 = go.Scatter(
        x=df[x_col],
        y=df[y2_col],
        mode='lines+markers',
        name=f'{y2_col}',
        line=dict(color='orange', width=2),
        marker=dict(size=8)
    )

    # Membuat layout untuk plot
    layout = go.Layout(
        title=dict(text=title, x=0.5, font=dict(size=20, color='darkblue')),
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label),
        showlegend=True,
        legend=dict(font=dict(size=12), x=0, y=1, xanchor='left', yanchor='top'),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='closest',
        plot_bgcolor='white',
        xaxis_gridcolor='lightgray',
        yaxis_gridcolor='lightgray',
        shapes=[
            # Garis putus-putus merah di y=0.5
            dict(
                type="line",
                x0=df[x_col].min(), x1=df[x_col].max(),
                y0=0.5, y1=0.5,
                line=dict(color="red", width=1, dash="dash")
            )
        ]
    )

    # Membuat figure dari trace dan layout
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Menampilkan plot di Streamlit
    st.plotly_chart(fig, use_container_width=True)

    
st.set_page_config(layout="wide")

def add_min_width_css():
    st.markdown(
        """
        <style>
        /* Set a minimum width for the app */
        .css-1d391kg {
            min-width: 3000px; /* Set the minimum width */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Add CSS styling to the app
add_min_width_css()

def download_file_from_github(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved to {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def load_excel(file_path):
    with open(file_path, 'rb') as file:
        model = pd.read_excel(file, engine='openpyxl')
    return model

def list_files_in_directory(dir_path):
    # Fungsi untuk mencetak semua isi dari suatu direktori
    for root, dirs, files in os.walk(dir_path):
        st.write(f'Direktori: {root}')
        for file_name in files:
            st.write(f'  - {file_name}')



st.title('Dashboard - Safety Stock')

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Fungsi untuk mereset state button
def reset_button_state():
    st.session_state.button_clicked = False


if 'df_month' not in locals():
    df_month = pd.read_csv('Dataset/Safety Stock/2024/df_month.csv')
    df_quarter = pd.read_csv('Dataset/Safety Stock/2024/df_quarter.csv')

list_bulan = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December']


def format_number(x):
    if x==0:
        return ''
    if isinstance(x, (int, float)):
        return "{:,.0f}".format(x)
    return x

def highlight_indikator(val):
    if val == 'OK':
        color = 'background-color: green; color: green;'
    elif val == 'OVER':
        color = 'background-color: red; color: red;'
    elif val == 'LOW':
        color = 'background-color: yellow; color: yellow;'
    else:
        color = ''
    return color
    



def create_dual_axis_chart(data, x_column, y_bar_column, y_line_column, title):
    fig = go.Figure()

    # Menambahkan bar chart
    fig.add_trace(
        go.Bar(
            x=data[x_column],
            y=data[y_bar_column],
            name=y_bar_column,
            marker_color='orange',
            yaxis='y1'
        )
    )

    # Menambahkan line chart
    fig.add_trace(
        go.Scatter(
            x=data[x_column],
            y=data[y_line_column],
            name=y_line_column,
            mode='lines+markers',
            line=dict(color='blue', width=4),
            yaxis='y2'
        )
    )

    # Menyesuaikan layout untuk dua sumbu y
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_column),
        yaxis=dict(
            title=y_bar_column,
            #titlefont=dict(color="orange"),
            tickfont=dict(color="orange")
        ),
        yaxis2=dict(
            title=y_line_column,
            #titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.1, y=1.1, orientation='h'),
        template="plotly_white"
    )

    return fig

quarter = st.selectbox("QUARTER:", ['Q1','Q2','Q3','Q4'], index=0, on_change=reset_button_state)
kategori = st.selectbox("KATEGORI:", ['SPAREPART','MAINTENANCE & REPAIR'], index=0, on_change=reset_button_state)
if 'Q1' == quarter:
    bulan = ['January','February','March']
if 'Q2' == quarter:
    bulan = ['April','May','June']
if 'Q3' == quarter:
    bulan = ['July','August','September']
if 'Q4' == quarter:
    bulan = ['October','November','December']


df_month['Month'] = pd.Categorical(df_month['Month'],categories=list_bulan)#pd.to_datetime(df_month['Month'],format='%B').sort_values().dt.strftime('%B').unique())
df_month = df_month[df_month['Kategori'].fillna('')==('MAINTENANCE' if kategori == 'MAINTENANCE & REPAIR' else '')].drop(columns='Kategori').sort_values(['Month','Nama Barang'])
df_quarter =df_quarter[df_quarter['Kategori'].fillna('')==('MAINTENANCE' if kategori == 'MAINTENANCE & REPAIR' else '')].drop(columns='Kategori').sort_values('Nama Barang')


df_3m = df_month[df_month['Month'].isin(bulan)].pivot(index='Nama Barang', columns=['Month'], values='AVG PICK UP').reset_index().merge(
    df_quarter[df_quarter['Quarter']==quarter][['Nama Barang','AVG PICK UP']],how='left').merge(
        df_month[df_month['Month'].isin(bulan)].pivot(index='Nama Barang', columns=['Month'], values='AVG PEMBELIAN').reset_index().merge(
            df_quarter[df_quarter['Quarter']==quarter][['Nama Barang','AVG PEMBELIAN','HARGA PEMBELIAN TERAKHIR','TOTAL [PEMBELIAN]']],how='left'), 
            how='left', on='Nama Barang').merge(df_month[df_month['Month'].isin(bulan)].pivot(index='Nama Barang', columns=['Month'], values='AVG SALDO AKHIR').reset_index().merge(
    df_quarter[df_quarter['Quarter']==quarter][['Nama Barang','AVG SALDO AKHIR']],how='left'), how='left')

df_3m.columns = [col.replace('_x', ' ') if col.endswith('_x') else col for col in df_3m.columns]
df_3m.columns = [col.replace('_y', '  ') if col.endswith('_y') else col for col in df_3m.columns]


st.dataframe(df_quarter[df_quarter['Quarter']==quarter].drop(columns='Quarter').style.format(lambda x: format_number(x)).applymap(highlight_indikator, subset=['OVERSTOCK','OVERSTOCK [ANGKA STANDART BARU]','OVERSTOCK [ANGKA STANDART LAMA]']), use_container_width=True, hide_index=True)
st.write('')
st.dataframe(df_3m.style.format(lambda x: format_number(x)), use_container_width=True, hide_index=True)

df_line = df_quarter[df_quarter['OVERSTOCK']=='OVER'].groupby('Quarter').agg({'Nama Barang':'count','TOTAL [PEMBELIAN]':'sum'}).rename(columns={'Nama Barang':'TOTAL BARANG','TOTAL [PEMBELIAN]':'TOTAL NOMINAL'}).reset_index()
fig = create_dual_axis_chart(df_line, 'Quarter', 'TOTAL NOMINAL', 'TOTAL BARANG','OVERSTOCK (QUARTER)')
st.plotly_chart(fig, use_container_width=True)

df_line2 = df_quarter[df_quarter['OVERSTOCK [ANGKA STANDART BARU]']=='OVER'].groupby('Quarter').agg({'Nama Barang':'count','TOTAL [SALDO AKHIR]':'sum'}).rename(columns={'Nama Barang':'TOTAL BARANG','TOTAL [SALDO AKHIR]':'TOTAL NOMINAL'}).reset_index()
fig = create_dual_axis_chart(df_line2, 'Quarter', 'TOTAL NOMINAL', 'TOTAL BARANG','OVERSTOCK [ANGKA STANDART BARU] (QUARTER)')
st.plotly_chart(fig, use_container_width=True)

df_line3 = df_quarter[df_quarter['OVERSTOCK [ANGKA STANDART LAMA]']=='OVER'].groupby('Quarter').agg({'Nama Barang':'count','TOTAL [SALDO AKHIR]':'sum'}).rename(columns={'Nama Barang':'TOTAL BARANG','TOTAL [SALDO AKHIR]':'TOTAL NOMINAL'}).reset_index()
fig = create_dual_axis_chart(df_line3, 'Quarter', 'TOTAL NOMINAL', 'TOTAL BARANG','OVERSTOCK [ANGKA STANDART LAMA] (QUARTER)')
st.plotly_chart(fig, use_container_width=True)
