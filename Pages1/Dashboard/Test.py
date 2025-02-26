import streamlit as st
import requests
import zipfile
import io
import pandas as pd
import os
import gdown
import tempfile

import plotly.express as px
import plotly.graph_objs as go
import streamlit as st

st.set_page_config(layout="wide")

st.markdown(
    """
    <div style="background-color: #68041c; padding: 10px; border-radius: 5px; text-align: center;">
        <h2 style="color: white; margin: 0;">Dashboard-Leadtime</h2>
    </div>
    """,
    unsafe_allow_html=True
)
st.write('')

def download_file_from_github(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved to {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Fungsi untuk mereset state button
def reset_button_state():
    st.session_state.button_clicked = False

if 'df_internal' not in locals():
  with zipfile.ZipFile(f'Dataset/Leadtime Internal & Eksternal/2025/Leadtime.zip', 'r') as z:
    with z.open('Leadtime_internal.csv') as f:
        df_internal = pd.read_csv(f)
    with z.open('Leadtime_eksternal_1.csv') as f:
        df_eksternal = pd.read_csv(f)
    with z.open('Leadtime_eksternal_2.csv') as f:
        df_eksternal2 = pd.read_csv(f)
    with z.open('Leadtime_resto.csv') as f:
        df_resto = pd.read_csv(f).fillna(0)


def create_pie_chart(df, labels_column, values_column, title="Pie Chart", key=None):
    color_mapping = {
        'On-Time': px.colors.sequential.RdBu[1],  
        'Backdate': px.colors.sequential.RdBu[0] 
    }
    fig = px.pie(
        df, 
        names=labels_column, 
        values=values_column, 
        title='',
        hole=0.3,
        color=labels_column,
        color_discrete_map=color_mapping  # Skema warna yang lebih estetis
    )
    
    # Kustomisasi tampilan
    fig.update_traces(
        textinfo='percent+label',  # Menampilkan label dan persentase
        textfont_size=12,  # Ukuran teks yang lebih besar
        pull=[0.1 if value == df[values_column].max() else 0 for value in df[values_column]]  # Memperbesar bagian terbesar
    )

    fig.update_layout(
        showlegend=True,  # Menampilkan legenda
        legend=dict(
            orientation="h",  # Menampilkan legenda secara horizontal
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    fig.update_layout(width=350, height=350)
    # Menampilkan grafik di Streamlit
    st.plotly_chart(fig, key=key)

def create_line_chart(df, x_column, y_column, title="Line Chart", key=None):
    """
    Membuat grafik garis (line chart) dengan Plotly dan menampilkannya di Streamlit.
    
    Parameters:
    - df: DataFrame yang berisi data untuk grafik
    - x_column: Nama kolom yang ingin digunakan sebagai sumbu X
    - y_column: Nama kolom yang ingin digunakan sebagai sumbu Y
    - title: Judul grafik (default: "Line Chart")
    """
    # Membuat line chart menggunakan Plotly
    fig = px.line(df, x=x_column, y=y_column)
    color = px.colors.sequential.RdBu[0]
    # Kustomisasi tampilan
    fig.update_traces(
        line=dict(color=color, width=2.5),  # Warna garis biru dengan ketebalan 2.5
        mode='lines+markers'  # Menampilkan garis dan titik data
    )
    
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column
    )
    fig.update_layout(width=500, height=350)
    # Menampilkan grafik di Streamlit
    st.plotly_chart(fig, key=key)

def create_percentage_barchart(df, x_col, y_col, key=None):
    # Menghitung persentase berdasarkan y_col
    df['Percentage'] = (df[y_col] / df[y_col].sum()) * 100

    # Membuat bar chart menggunakan Plotly
    fig = px.bar(
        df,
        x=x_col,
        y='Percentage',
        labels={x_col: 'Kategori', 'Percentage': '%'},
        color_discrete_sequence=px.colors.sequential.RdBu,
        hover_data={y_col: True, 'Percentage': ':.2f%'}  # Menampilkan angka kuantitas dan persentase pada hover
    )
    
    # Menambahkan nilai persentase pada setiap bar
    fig.update_traces(
        texttemplate='%{y:.2f}%', 
        textposition='inside', 
        insidetextanchor='middle'
    )
    
    # Mengatur layout grafik
    fig.update_layout(
        width=350, 
        height=350
    )
    
    # Menampilkan grafik di Streamlit
    st.plotly_chart(fig, key=key)

        
df_internal['Rute Global'] = pd.Categorical(df_internal['Rute Global'],['WH/DC to WH/DC','WH/DC to Resto','Resto to WH/DC','Resto to Resto'])
df_eksternal['Tanggal PO'] = pd.to_datetime(df_eksternal['Tanggal PO'])
df_eksternal2['Tanggal PO'] = pd.to_datetime(df_eksternal2['Tanggal PO'])
df_internal['Tanggal IT Kirim'] = pd.to_datetime(df_internal['Tanggal IT Kirim'])
df_internal['Tanggal IT Terima'] = pd.to_datetime(df_internal['Tanggal IT Terima'])
df_internal['Tanggal Kirim'] = pd.to_datetime(df_internal['Bulan Kirim'],format='%b-%y')
df_internal['Tanggal Terima'] = pd.to_datetime(df_internal['Bulan Terima'],format='%b-%y')
df_eksternal['Date PO'] = pd.to_datetime(df_eksternal['Bulan PO'],format='%b-%y')
df_eksternal2['Date PO'] = pd.to_datetime(df_eksternal2['Bulan PO'],format='%b-%y')

label_kategori = ['(-)','(<=0)','(1)','(2-3)','(4-7)','(>7)']
df_eksternal['PR(Create)-PO(Datang) Group'] = pd.Categorical(df_eksternal['PR(Create)-PO(Datang) Group'],categories=label_kategori)
df_eksternal['PO(Create)-PO(Datang) Group'] = pd.Categorical(df_eksternal['PO(Create)-PO(Datang) Group'],categories=label_kategori)
df_eksternal['RI(Date)-PO(Datang) Group'] = pd.Categorical(df_eksternal['RI(Date)-PO(Datang) Group'],categories=label_kategori)
df_eksternal2['RI(Date)-RI(Create) Group'] = pd.Categorical(df_eksternal2['RI(Date)-RI(Create) Group'],categories=label_kategori)


# Arahkan ke aplikasi berdasarkan pilihan pengguna
internal_tab, eksternal_tab, = st.tabs(["Leadtime-Internal", "Leadtime-Eksternal"])
with internal_tab:
    def style_table():
        html = f"""
        <style>
            table {{
                font-size: 12px;  /* Ganti ukuran sesuai kebutuhan */
                width: 100%;
            }}
        </style>
        """
        return html
    st.markdown(style_table(), unsafe_allow_html=True)
        
    def highlight_first_word(df, col_name):
        def apply_highlight(value):
            # Pisahkan kata-kata dalam cell
            words = value.split()
            # Warna hanya kata pertama
            if words:
                words[0] = f"<span style='color: red;'>{words[0]}</span>"
            return " ".join(words)
    
        # Terapkan fungsi highlight pada kolom yang dimaksud
        styled_df = df.copy()
        styled_df[col_name] = styled_df[col_name].apply(apply_highlight)
        return styled_df
    
    def highlight_last_word(df, col_name):
        def apply_highlight(value):
            # Pisahkan kata-kata dalam cell
            words = value.split()
            # Warna hanya kata pertama
            if words:
                words[-1] = f"<span style='color: red;'>{words[-1]}</span>"
            return " ".join(words)
    
        # Terapkan fungsi highlight pada kolom yang dimaksud
        styled_df = df.copy()
        styled_df[col_name] = styled_df[col_name].apply(apply_highlight)
        return styled_df
        
    st.markdown(
        """
        <div style="background-color: #68041c; padding: 10px; border-radius: 5px; text-align: center;">
            <h3 style="color: white; margin: 0;">Leadtime-Internal</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
        
    st.write(' ')
    
    st.markdown('## Summary')
    df_resto = df_resto.set_index('Month')
    #st.dataframe(df_resto.style.format(lambda x: '' if x == 0 else '{:,.0f}'.format(x)))#.apply(lambda x: ['background: red' if i == 0 else '' for i in range(len(x))], axis=1).background_gradient(cmap='Reds', axis=1, subset=df_resto.iloc[0:1, :]))    
    st.dataframe(df_resto.style.format(lambda x: '' if x == 0 else '{:,.0f}'.format(x)).background_gradient(cmap='Reds', axis=0, subset=['CTP']))

    st.markdown('### Outgoing Backdate')
    col = st.columns([1,2,1,2])
    with col[0]:
        df_pie = df_internal.groupby(['Kategori Leadtime SJ'])[['Nomor IT Kirim']].nunique().reset_index()
        
        create_pie_chart(df_pie, labels_column='Kategori Leadtime SJ', values_column='Nomor IT Kirim', title="OUTGOING BACKDATE")
    with col[1]:
        df_line = df_internal.groupby(['Tanggal Kirim','Kategori Leadtime SJ'])[['Nomor IT Kirim']].nunique().reset_index().groupby('Tanggal Kirim')[['Nomor IT Kirim']].sum().reset_index().rename(columns={'Nomor IT Kirim':'Total'}).merge(
            df_internal[df_internal['Kategori Leadtime SJ']=='Backdate'].groupby(['Tanggal Kirim'])[['Nomor IT Kirim']].nunique().reset_index(), how='left'
        )
        df_line['Nomor IT Kirim'] = (df_line['Nomor IT Kirim']/df_line['Total'])*100
        create_line_chart(df_line, x_column='Tanggal Kirim', y_column='Nomor IT Kirim', title="DAILY BACKDATE")
    with col[2]:
        df_bar = df_internal[
                 (df_internal['Kategori Leadtime SJ']=='Backdate')].groupby(['Kirim #2'])[['Nomor IT Kirim']].nunique().reset_index()
        create_percentage_barchart(df_bar, 'Kirim #2', 'Nomor IT Kirim')
    with col[3]:
        st.write('')
        col2 = st.columns(3)
        with col2[0]:
            st.metric(label="Total", value="{:,.0f}".format(df_internal['Nomor IT Kirim'].nunique()), delta=None)
        with col2[1]:
            st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori Leadtime SJ']=='On-Time']['Nomor IT Kirim'].values[0]), delta=None)
        with col2[2]:
            st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori Leadtime SJ']=='Backdate']['Nomor IT Kirim'].values[0]), delta=None)
        
        df_tabel = df_internal[
                 (df_internal['Kategori Leadtime SJ']=='Backdate')].groupby(['Leadtime SJ Group','Rute Global'])[['Nomor IT Kirim']].nunique().reset_index().pivot(index='Rute Global',columns='Leadtime SJ Group',values='Nomor IT Kirim').reset_index().merge(
            df_internal[
                 (df_internal['Kategori Leadtime SJ']=='Backdate')].groupby(['Rute Global'])[['Nomor IT Kirim']].nunique().reset_index().rename(columns={'Nomor IT Kirim':'Total'})
                 )
        
        df_tabel = pd.concat([df_tabel,pd.concat([df_internal[(df_internal['Kategori Leadtime SJ']=='Backdate')].groupby(['Leadtime SJ Group'])[['Nomor IT Kirim']].nunique().rename(columns={'Nomor IT Kirim':'Total'}),
                               pd.DataFrame([df_internal[(df_internal['Kategori Leadtime SJ']=='Backdate')][['Nomor IT Kirim']].nunique().values],['Total'],columns=['Total'])]).T.reset_index().rename(columns={'index':'Rute Global'})])
        styled_df = highlight_first_word(df_tabel, "Rute Global")
        st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)


    
    st.markdown('### Incoming Backdate')
    col = st.columns([1,2,1,2])
    with col[0]:
        df_pie = df_internal.groupby(['Kategori Leadtime RI'])[['Nomor IT Terima']].nunique().reset_index()
        
        create_pie_chart(df_pie, labels_column='Kategori Leadtime RI', values_column='Nomor IT Terima', title="INCOMING BACKDATE")
    with col[1]:
        df_line = df_internal.groupby(['Tanggal Terima','Kategori Leadtime RI'])[['Nomor IT Terima']].nunique().reset_index().groupby('Tanggal Terima')[['Nomor IT Terima']].sum().reset_index().rename(columns={'Nomor IT Terima':'Total'}).merge(
            df_internal[df_internal['Kategori Leadtime RI']=='Backdate'].groupby(['Tanggal Terima'])[['Nomor IT Terima']].nunique().reset_index(), how='left'
        )
        df_line['Nomor IT Terima'] = (df_line['Nomor IT Terima']/df_line['Total'])*100
        create_line_chart(df_line, x_column='Tanggal Terima', y_column='Nomor IT Terima', title="DAILY BACKDATE")
    with col[2]:
        df_bar = df_internal[
                 (df_internal['Kategori Leadtime RI']=='Backdate')].groupby(['Terima #2'])[['Nomor IT Terima']].nunique().reset_index()
        create_percentage_barchart(df_bar, 'Terima #2', 'Nomor IT Terima')
    with col[3]:
        st.write('')
        col2 = st.columns(3)
        with col2[0]:
            st.metric(label="Total", value="{:,.0f}".format(df_internal['Nomor IT Terima'].nunique()), delta=None)
        with col2[1]:
            st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori Leadtime RI']=='On-Time']['Nomor IT Terima'].values[0]), delta=None)
        with col2[2]:
            st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori Leadtime RI']=='Backdate']['Nomor IT Terima'].values[0]), delta=None)
    
        df_tabel = df_internal[
                  (df_internal['Kategori Leadtime RI']=='Backdate')].groupby(['Leadtime RI Group','Rute Global'])[['Nomor IT Terima']].nunique().reset_index().pivot(index='Rute Global',columns='Leadtime RI Group',values='Nomor IT Terima').reset_index().merge(
            df_internal[
                  (df_internal['Kategori Leadtime RI']=='Backdate')].groupby(['Rute Global'])[['Nomor IT Terima']].nunique().reset_index().rename(columns={'Nomor IT Terima':'Total'})
                 )
        df_tabel = pd.concat([df_tabel,pd.concat([df_internal[(df_internal['Kategori Leadtime RI']=='Backdate')].groupby(['Leadtime RI Group'])[['Nomor IT Terima']].nunique().rename(columns={'Nomor IT Terima':'Total'}),
                               pd.DataFrame([df_internal[(df_internal['Kategori Leadtime RI']=='Backdate')][['Nomor IT Terima']].nunique().values],['Total'],columns=['Total'])]).T.reset_index().rename(columns={'index':'Rute Global'})])
        styled_df = highlight_last_word(df_tabel, "Rute Global")
        st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    data = {
        "Tanggal": pd.date_range(start="2024-01-01", end="2025-12-31", freq="D"),
        "Penjualan": range(1, 732)
    }
    df = pd.DataFrame(data)
    
    
    # Widget untuk memilih rentang tanggal
    start_date, end_date = st.date_input(
        "RANGE DATE: ",
        [df["Tanggal"].min(), df["Tanggal"].max()],  # Default nilai awal
        min_value=df["Tanggal"].min(),
        max_value=df["Tanggal"].max()
    )
    
    pic = st.selectbox("PIC RESPONSIBLE:", ['All','WH/DC','Resto'], index=0, on_change=reset_button_state)
    
    st.markdown('### Outgoing Backdate')
    col = st.columns([1,2,1,2])
    with col[0]:
        df_pie = df_internal[(df_internal["Tanggal IT Kirim"] >= pd.Timestamp(start_date)) & (df_internal["Tanggal IT Kirim"] <= pd.Timestamp(end_date)) & (df_internal['Kirim #2'].isin(['Resto','WH/DC'] if pic=='All' else [pic]))].groupby(['Kategori Leadtime SJ'])[['Nomor IT Kirim']].nunique().reset_index()
        
        create_pie_chart(df_pie, labels_column='Kategori Leadtime SJ', values_column='Nomor IT Kirim', title="OUTGOING BACKDATE2", key='pie')
    with col[1]:
        df_line = df_internal[(df_internal["Tanggal IT Kirim"] >= pd.Timestamp(start_date)) & (df_internal["Tanggal IT Kirim"] <= pd.Timestamp(end_date)) & (df_internal['Kirim #2'].isin(['Resto','WH/DC'] if pic=='All' else [pic]))
                     & (df_internal['Kategori Leadtime SJ']=='Backdate')].groupby(['Tanggal IT Kirim'])[['Nomor IT Kirim']].nunique().reset_index()
        
        create_line_chart(df_line, x_column='Tanggal IT Kirim', y_column='Nomor IT Kirim', title="DAILY BACKDATE", key='line')
    with col[2]:
        df_bar = df_internal[(df_internal["Tanggal IT Kirim"] >= pd.Timestamp(start_date)) & (df_internal["Tanggal IT Kirim"] <= pd.Timestamp(end_date)) & (df_internal['Kirim #2'].isin(['Resto','WH/DC'] if pic=='All' else [pic]))
                 & (df_internal['Kategori Leadtime SJ']=='Backdate')].groupby(['Kirim #2'])[['Nomor IT Kirim']].nunique().reset_index()
        create_percentage_barchart(df_bar, 'Kirim #2', 'Nomor IT Kirim', key='bar')
    with col[3]:
        st.write('')
        col2 = st.columns(3)
        with col2[0]:
            st.metric(label="Total", value="{:,.0f}".format(df_internal[(df_internal["Tanggal IT Terima"] >= pd.Timestamp(start_date)) & (df_internal["Tanggal IT Terima"] <= pd.Timestamp(end_date)) & (df_internal['Kirim #2'].isin(['Resto','WH/DC'] if pic=='All' else [pic]))]['Nomor IT Kirim'].nunique()), delta=None)
        with col2[1]:
            st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori Leadtime SJ']=='On-Time']['Nomor IT Kirim'].values[0]), delta=None)
        with col2[2]:
            st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori Leadtime SJ']=='Backdate']['Nomor IT Kirim'].values[0]), delta=None)
        
        df_tabl = df_internal[(df_internal["Tanggal IT Kirim"] >= pd.Timestamp(start_date)) & (df_internal["Tanggal IT Kirim"] <= pd.Timestamp(end_date)) & (df_internal['Kirim #2'].isin(['Resto','WH/DC'] if pic=='All' else [pic]))
                 & (df_internal['Kategori Leadtime SJ']=='Backdate')]
        df_tabel = df_tabl.groupby(['Leadtime SJ Group','Rute Global'])[['Nomor IT Kirim']].nunique().reset_index().pivot(index='Rute Global',columns='Leadtime SJ Group',values='Nomor IT Kirim').reset_index().merge(
            df_tabl.groupby(['Rute Global'])[['Nomor IT Kirim']].nunique().reset_index().rename(columns={'Nomor IT Kirim':'Total'})
                 )

        if pic =='Resto':
            df_tabel = df_tabel.loc[df_tabel['Rute Global'].isin(['Resto to WH/DC','Resto to Resto'])]
            styled_df = highlight_first_word(df_tabel, "Rute Global")
            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        elif pic=='WH/DC':
            df_tabel = df_tabel.loc[df_tabel['Rute Global'].isin(['WH/DC to WH/DC','WH/DC to Resto'])]
            styled_df = highlight_first_word(df_tabel, "Rute Global")
            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            df_tabel = pd.concat([df_tabel,pd.concat([df_tabl.groupby(['Leadtime SJ Group'])[['Nomor IT Kirim']].nunique().rename(columns={'Nomor IT Kirim':'Total'}),
                            pd.DataFrame([df_tabl[['Nomor IT Kirim']].nunique().values],['Total'],columns=['Total'])]).T.reset_index().rename(columns={'index':'Rute Global'})])
            styled_df = highlight_first_word(df_tabel, "Rute Global")
            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    
    st.markdown('### Incoming Backdate')
    col = st.columns([1,2,1,2])
    with col[0]:
        df_pie = df_internal[(df_internal["Tanggal IT Terima"] >= pd.Timestamp(start_date)) & (df_internal["Tanggal IT Terima"] <= pd.Timestamp(end_date)) & (df_internal['Terima #2'].isin(['Resto','WH/DC'] if pic=='All' else [pic]))].groupby(['Kategori Leadtime RI'])[['Nomor IT Terima']].nunique().reset_index()
        
        create_pie_chart(df_pie, labels_column='Kategori Leadtime RI', values_column='Nomor IT Terima', title="INCOMING BACKDATE2",key='pie2')
    with col[1]:
        df_line = df_internal[(df_internal["Tanggal IT Terima"] >= pd.Timestamp(start_date)) & (df_internal["Tanggal IT Terima"] <= pd.Timestamp(end_date)) & (df_internal['Terima #2'].isin(['Resto','WH/DC'] if pic=='All' else [pic]))
                     & (df_internal['Kategori Leadtime RI']=='Backdate')].groupby(['Tanggal IT Terima'])[['Nomor IT Terima']].nunique().reset_index()
        
        create_line_chart(df_line, x_column='Tanggal IT Terima', y_column='Nomor IT Terima', title="DAILY BACKDATE", key='line2')
    with col[2]:
        df_bar = df_internal[(df_internal["Tanggal IT Terima"] >= pd.Timestamp(start_date)) & (df_internal["Tanggal IT Terima"] <= pd.Timestamp(end_date)) & (df_internal['Terima #2'].isin(['Resto','WH/DC'] if pic=='All' else [pic]))
                 & (df_internal['Kategori Leadtime RI']=='Backdate')].groupby(['Terima #2'])[['Nomor IT Terima']].nunique().reset_index()
        create_percentage_barchart(df_bar, 'Terima #2', 'Nomor IT Terima', key='bar2')
    with col[3]:
        st.write('')
        col2 = st.columns(3)
        with col2[0]:
            st.metric(label="Total", value="{:,.0f}".format(df_internal[(df_internal["Tanggal IT Terima"] >= pd.Timestamp(start_date)) & (df_internal["Tanggal IT Terima"] <= pd.Timestamp(end_date)) & (df_internal['Terima #2'].isin(['Resto','WH/DC'] if pic=='All' else [pic]))]['Nomor IT Terima'].nunique()), delta=None)
        with col2[1]:
            st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori Leadtime RI']=='On-Time']['Nomor IT Terima'].values[0]), delta=None)
        with col2[2]:
            st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori Leadtime RI']=='Backdate']['Nomor IT Terima'].values[0]), delta=None)
    
        df_tabl = df_internal[(df_internal["Tanggal IT Terima"] >= pd.Timestamp(start_date)) & (df_internal["Tanggal IT Terima"] <= pd.Timestamp(end_date)) & (df_internal['Terima #2'].isin(['Resto','WH/DC'] if pic=='All' else [pic]))
                 & (df_internal['Kategori Leadtime RI']=='Backdate')]
        df_tabel = df_tabl.groupby(['Leadtime RI Group','Rute Global'])[['Nomor IT Terima']].nunique().reset_index().pivot(index='Rute Global',columns='Leadtime RI Group',values='Nomor IT Terima').reset_index().merge(
                df_tabl.groupby(['Rute Global'])[['Nomor IT Terima']].nunique().reset_index().rename(columns={'Nomor IT Terima':'Total'})
                 )
        
        if pic=='Resto':
            df_tabel = df_tabel.loc[df_tabel['Rute Global'].isin(['WH/DC to Resto','Resto to Resto'])]
            styled_df = highlight_last_word(df_tabel, "Rute Global")
            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        elif pic=='WH/DC':
            df_tabel = df_tabel.loc[df_tabel['Rute Global'].isin(['Resto to WH/DC','WH/DC to WH/DC'])]
            styled_df = highlight_last_word(df_tabel, "Rute Global")
            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            df_tabel = pd.concat([df_tabel,pd.concat([df_tabl.groupby(['Leadtime RI Group'])[['Nomor IT Terima']].nunique().rename(columns={'Nomor IT Terima':'Total'}),
                            pd.DataFrame([df_tabl[['Nomor IT Terima']].nunique().values],['Total'],columns=['Total'])]).T.reset_index().rename(columns={'index':'Rute Global'})])
            styled_df = highlight_last_word(df_tabel, "Rute Global")
            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
with eksternal_tab:
    st.markdown(
        """
        <div style="background-color: #68041c; padding: 10px; border-radius: 5px; text-align: center;">
            <h3 style="color: white; margin: 0;">Leadtime-Eksternal</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write('')
    st.markdown('## Summary')
    kat_eksternal = st.selectbox("KATEGORI BACKDATE:", ['PR(Create)-PO(Datang)','PO(Create)-PO(Datang)','RI(Date)-PO(Datang)','RI(Date)-RI(Create)'], index=0, on_change=reset_button_state)
    
    if kat_eksternal =='PR(Create)-PO(Datang)':
        st.markdown('### PR(Create)-PO(Datang)')
        st.write('PIC Responsible: Logistic')
        st.write('Kategori Item: Eksternal Logistic')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori Item']=='Eksternal Logistic')].groupby(['Kategori PR(Create)-PO(Datang)'])[['Nomor PR+PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori PR(Create)-PO(Datang)', values_column='Nomor PR+PO', title="OUTGOING BACKDATE", key='pie_')
        with col[1]:
            df_line = df_eksternal[(df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori Item']=='Eksternal Logistic')
                     ].groupby(['Date PO'])[['Nomor PR+PO']].nunique().reset_index().rename(columns={'Nomor PR+PO':'Total'}).merge(df_eksternal[(df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Logistic')
                     ].groupby(['Date PO'])[['Nomor PR+PO']].nunique().reset_index(), how='left')
            df_line['Nomor PR+PO'] = (df_line['Nomor PR+PO']/df_line['Total'])*100
            create_line_chart(df_line, x_column='Date PO', y_column='Nomor PR+PO', title="DAILY BACKDATE",key='line_')
        with col[2]:
            df_bar = df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Logistic')].groupby(['Rute'])[['Nomor PR+PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PR+PO', key='bar_')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori Item']=='Eksternal Logistic')]['Nomor PR+PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori PR(Create)-PO(Datang)']=='On-Time']['Nomor PR+PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori PR(Create)-PO(Datang)']=='Backdate']['Nomor PR+PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Logistic')].groupby(['Rute','PR(Create)-PO(Datang) Group'])[['Nomor PR+PO']].nunique().reset_index().pivot(index='Rute',columns='PR(Create)-PO(Datang) Group',values='Nomor PR+PO').reset_index().merge(
                df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Logistic')].groupby(['Rute'])[['Nomor PR+PO']].nunique().reset_index().rename(columns={'Nomor PR+PO':'Total'})
                     ),
                         hide_index=True)
        
        st.write('Kategori Item: Internal Logistic')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori Item']=='Internal Logistic')].groupby(['Kategori PR(Create)-PO(Datang)'])[['Nomor PR+PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori PR(Create)-PO(Datang)', values_column='Nomor PR+PO', title="OUTGOING BACKDATE",key='pie_1')
        with col[1]:
            df_line = df_eksternal[(df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori Item']=='Internal Logistic')
                     ].groupby(['Date PO'])[['Nomor PR+PO']].nunique().reset_index().rename(columns={'Nomor PR+PO':'Total'}).merge(df_eksternal[(df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Internal Logistic')
                     ].groupby(['Date PO'])[['Nomor PR+PO']].nunique().reset_index(), how='left')
            df_line['Nomor PR+PO'] = (df_line['Nomor PR+PO']/df_line['Total'])*100
            create_line_chart(df_line, x_column='Date PO', y_column='Nomor PR+PO', title="DAILY BACKDATE",key='line_1')
        with col[2]:
            df_bar = df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Internal Logistic')].groupby(['Rute'])[['Nomor PR+PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PR+PO',key='bar_1')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori Item']=='Internal Logistic')]['Nomor PR+PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori PR(Create)-PO(Datang)']=='On-Time']['Nomor PR+PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori PR(Create)-PO(Datang)']=='Backdate']['Nomor PR+PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Internal Logistic')].groupby(['Rute','PR(Create)-PO(Datang) Group'])[['Nomor PR+PO']].nunique().reset_index().pivot(index='Rute',columns='PR(Create)-PO(Datang) Group',values='Nomor PR+PO').reset_index().merge(
                df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Internal Logistic')].groupby(['Rute'])[['Nomor PR+PO']].nunique().reset_index().rename(columns={'Nomor PR+PO':'Total'})
                     ),
                         hide_index=True)
            
        st.write('Kategori Item: Resto')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[ (df_eksternal['PIC Responsible']=='Resto') 
                     ].groupby(['Kategori PR(Create)-PO(Datang)'])[['Nomor PR+PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori PR(Create)-PO(Datang)', values_column='Nomor PR+PO', title="OUTGOING BACKDATE", key='pie_2')
        with col[1]:
            df_line = df_eksternal[(df_eksternal['PIC Responsible']=='Resto') 
                     ].groupby(['Date PO'])[['Nomor PR+PO']].nunique().reset_index().rename(columns={'Nomor PR+PO':'Total'}).merge(df_eksternal[(df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate')
                     ].groupby(['Date PO'])[['Nomor PR+PO']].nunique().reset_index(), how='left')
            df_line['Nomor PR+PO'] = (df_line['Nomor PR+PO']/df_line['Total'])*100
            create_line_chart(df_line, x_column='Date PO', y_column='Nomor PR+PO', title="DAILY BACKDATE", key='line_2')
        with col[2]:
            df_bar = df_eksternal[ (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PR+PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PR+PO', key='bar_2')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[ (df_eksternal['PIC Responsible']=='Resto') ]['Nomor PR+PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori PR(Create)-PO(Datang)']=='On-Time']['Nomor PR+PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori PR(Create)-PO(Datang)']=='Backdate']['Nomor PR+PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[ (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate')].groupby(['Rute','PR(Create)-PO(Datang) Group'])[['Nomor PR+PO']].nunique().reset_index().pivot(index='Rute',columns='PR(Create)-PO(Datang) Group',values='Nomor PR+PO').reset_index().merge(
                df_eksternal[ (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PR+PO']].nunique().reset_index().rename(columns={'Nomor PR+PO':'Total'})
                     ),
                         hide_index=True)
    
    if kat_eksternal =='PO(Create)-PO(Datang)':
        st.markdown('### Kategori PO(Create)-PO(Datang)')
        st.write('PIC Responsible: Procurement')
        st.write('Kategori Item: Logistic')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                     ].groupby(['Kategori PO(Create)-PO(Datang)'])[['Nomor PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori PO(Create)-PO(Datang)', values_column='Nomor PO', title="OUTGOING BACKDATE",key='pie_')
        with col[1]:
            df_line = df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic')
                     ].groupby(['Date PO'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'}).merge(df_eksternal[(df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate')
                     ].groupby(['Date PO'])[['Nomor PO']].nunique().reset_index(), how='left')
            df_line['Nomor PO'] = (df_line['Nomor PO']/df_line['Total'])*100
            create_line_chart(df_line, x_column='Date PO', y_column='Nomor PO', title="DAILY BACKDATE",key='line_')
        with col[2]:
            df_bar = df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') &
                     (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO', key='bar_')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                    ]['Nomor PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori PO(Create)-PO(Datang)']=='On-Time']['Nomor PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori PO(Create)-PO(Datang)']=='Backdate']['Nomor PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') &
                     (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate')].groupby(['Rute','PO(Create)-PO(Datang) Group'])[['Nomor PO']].nunique().reset_index().pivot(index='Rute',columns='PO(Create)-PO(Datang) Group',values='Nomor PO').reset_index().merge(
                df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') &
                     (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'})
                     ),
                         hide_index=True)
            
        st.write('Kategori Item: Resto')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[(df_eksternal['PIC Responsible']=='Resto') &
                     (df_eksternal['Kategori Item']=='Eksternal Resto')].groupby(['Kategori PO(Create)-PO(Datang)'])[['Nomor PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori PO(Create)-PO(Datang)', values_column='Nomor PO', title="OUTGOING BACKDATE", key='pie_1')
        with col[1]:
            df_line = df_eksternal[ (df_eksternal['PIC Responsible']=='Resto')
                     ].groupby(['Date PO'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'}).merge(df_eksternal[(df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate')
                     ].groupby(['Date PO'])[['Nomor PO']].nunique().reset_index(), how='left')
            df_line['Nomor PO'] = (df_line['Nomor PO']/df_line['Total'])*100
            create_line_chart(df_line, x_column='Date PO', y_column='Nomor PO', title="DAILY BACKDATE",key='line_1')
        with col[2]:
            df_bar = df_eksternal[ (df_eksternal['PIC Responsible']=='Resto') &
                     (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Resto')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO', key='bar_1')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[ (df_eksternal['PIC Responsible']=='Resto') 
                    ]['Nomor PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori PO(Create)-PO(Datang)']=='On-Time']['Nomor PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori PO(Create)-PO(Datang)']=='Backdate']['Nomor PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[ (df_eksternal['PIC Responsible']=='Resto') &
                     (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate')].groupby(['Rute','PO(Create)-PO(Datang) Group'])[['Nomor PO']].nunique().reset_index().pivot(index='Rute',columns='PO(Create)-PO(Datang) Group',values='Nomor PO').reset_index().merge(
                df_eksternal[ (df_eksternal['PIC Responsible']=='Resto') &
                     (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'})
                     ),
                         hide_index=True)
    
    if kat_eksternal =='RI(Date)-PO(Datang)':
        st.markdown('### Kategori RI(Date)-PO(Datang)')
        st.write('PIC Responsible: Procurement')
        st.write('Kategori Item: Logistic')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                     ].groupby(['Kategori RI(Date)-PO(Datang)'])[['Nomor PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori RI(Date)-PO(Datang)', values_column='Nomor PO', title="OUTGOING BACKDATE",key='pie_')
        with col[1]:
            df_line = df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic')
                     ].groupby(['Date PO'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'}).merge(df_eksternal[(df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate')
                     ].groupby(['Date PO'])[['Nomor PO']].nunique().reset_index(), how='left')
            df_line['Nomor PO'] = (df_line['Nomor PO']/df_line['Total'])*100
            create_line_chart(df_line, x_column='Date PO', y_column='Nomor PO', title="DAILY BACKDATE",key='line_')
        with col[2]:
            df_bar = df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') &
                     (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO', key='bar_')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') 
                    ]['Nomor PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-PO(Datang)']=='On-Time']['Nomor PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-PO(Datang)']=='Backdate']['Nomor PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') &
                     (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate')].groupby(['Rute','RI(Date)-PO(Datang) Group'])[['Nomor PO']].nunique().reset_index().pivot(index='Rute',columns='RI(Date)-PO(Datang) Group',values='Nomor PO').reset_index().merge(
                df_eksternal[ (df_eksternal['PIC Responsible']=='Logistic') &
                     (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'})
                     ),
                         hide_index=True)
            
        st.write('Kategori Item: Resto')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[(df_eksternal['PIC Responsible']=='Resto') &
                     (df_eksternal['Kategori Item']=='Eksternal Resto')].groupby(['Kategori RI(Date)-PO(Datang)'])[['Nomor PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori RI(Date)-PO(Datang)', values_column='Nomor PO', title="OUTGOING BACKDATE", key='pie_1')
        with col[1]:
            df_line = df_eksternal[ (df_eksternal['PIC Responsible']=='Resto')
                     ].groupby(['Date PO'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'}).merge(df_eksternal[(df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate')
                     ].groupby(['Date PO'])[['Nomor PO']].nunique().reset_index(), how='left')
            df_line['Nomor PO'] = (df_line['Nomor PO']/df_line['Total'])*100
            create_line_chart(df_line, x_column='Date PO', y_column='Nomor PO', title="DAILY BACKDATE",key='line_1')
        with col[2]:
            df_bar = df_eksternal[ (df_eksternal['PIC Responsible']=='Resto') &
                     (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Resto')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO', key='bar_1')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[ (df_eksternal['PIC Responsible']=='Resto') 
                    ]['Nomor PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-PO(Datang)']=='On-Time']['Nomor PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-PO(Datang)']=='Backdate']['Nomor PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[ (df_eksternal['PIC Responsible']=='Resto') &
                     (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate')].groupby(['Rute','RI(Date)-PO(Datang) Group'])[['Nomor PO']].nunique().reset_index().pivot(index='Rute',columns='RI(Date)-PO(Datang) Group',values='Nomor PO').reset_index().merge(
                df_eksternal[ (df_eksternal['PIC Responsible']=='Resto') &
                     (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'})
                     ),
                         hide_index=True)

    if kat_eksternal =='RI(Date)-RI(Create)':
        st.markdown('### Kategori RI(Date)-RI(Create)')
        st.write('PIC Responsible: Resto')
        st.write('Kategori Item: Eksternal Logistic')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal2[ 
                     (df_eksternal2['Kategori Item']=='Eksternal Logistic')].groupby(['Kategori RI(Date)-RI(Create)'])[['Nomor PO+RI']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori RI(Date)-RI(Create)', values_column='Nomor PO+RI', title="OUTGOING BACKDATE", key='pie_')
        with col[1]:
            df_line = df_eksternal2[(df_eksternal2['Kategori Item']=='Eksternal Logistic')
                     ].groupby(['Date PO'])[['Nomor PO+RI']].nunique().reset_index().rename(columns={'Nomor PO+RI':'Total'}).merge(df_eksternal2[ 
                      (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Eksternal Logistic')
                     ].groupby(['Date PO'])[['Nomor PO+RI']].nunique().reset_index(), how='left')
            df_line['Nomor PO+RI'] = (df_line['Nomor PO+RI']/df_line['Total'])*100
            create_line_chart(df_line, x_column='Date PO', y_column='Nomor PO+RI', title="DAILY BACKDATE",key='line_')
        with col[2]:
            df_bar = df_eksternal2[
                     (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Eksternal Logistic')].groupby(['Rute'])[['Nomor PO+RI']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO+RI',key='bar_')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal2[ (df_eksternal2['PIC Responsible']=='Logistic') 
                     & (df_eksternal2['Kategori Item']=='Eksternal Logistic')]['Nomor PO+RI'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-RI(Create)']=='On-Time']['Nomor PO+RI'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-RI(Create)']=='Backdate']['Nomor PO+RI'].values[0]), delta=None)
            
            st.dataframe(df_eksternal2[ (df_eksternal2['PIC Responsible']=='Logistic') &
                     (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Eksternal Logistic')].groupby(['Rute','RI(Date)-RI(Create) Group'])[['Nomor PO+RI']].nunique().reset_index().pivot(index='Rute',columns='RI(Date)-RI(Create) Group',values='Nomor PO+RI').reset_index().merge(
                df_eksternal2[ (df_eksternal2['PIC Responsible']=='Logistic') &
                     (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Eksternal Logistic')].groupby(['Rute'])[['Nomor PO+RI']].nunique().reset_index().rename(columns={'Nomor PO+RI':'Total'})
                     ),
                         hide_index=True)
            
        st.write('Kategori Item: Eksternal Resto')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal2[ 
                      (df_eksternal2['Kategori Item']=='Eksternal Resto')].groupby(['Kategori RI(Date)-RI(Create)'])[['Nomor PO+RI']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori RI(Date)-RI(Create)', values_column='Nomor PO+RI', title="OUTGOING BACKDATE",key='pie_1')
        with col[1]:
            df_line = df_eksternal2[(df_eksternal2['Kategori Item']=='Eksternal Resto')
                     ].groupby(['Date PO'])[['Nomor PO+RI']].nunique().reset_index().rename(columns={'Nomor PO+RI':'Total'}).merge(df_eksternal2[ 
                      (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Eksternal Resto')
                     ].groupby(['Date PO'])[['Nomor PO+RI']].nunique().reset_index(), how='left')
            df_line['Nomor PO+RI'] = (df_line['Nomor PO+RI']/df_line['Total'])*100
            create_line_chart(df_line, x_column='Date PO', y_column='Nomor PO+RI', title="DAILY BACKDATE",key='line_1')
        with col[2]:
            df_bar = df_eksternal2[
                     (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Eksternal Resto')].groupby(['Rute'])[['Nomor PO+RI']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO+RI',key='bar_1')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal2[  
                     (df_eksternal2['Kategori Item']=='Eksternal Resto')]['Nomor PO+RI'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-RI(Create)']=='On-Time']['Nomor PO+RI'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-RI(Create)']=='Backdate']['Nomor PO+RI'].values[0]), delta=None)
            
            st.dataframe(df_eksternal2[
                     (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Eksternal Resto')].groupby(['Rute','RI(Date)-RI(Create) Group'])[['Nomor PO+RI']].nunique().reset_index().pivot(index='Rute',columns='RI(Date)-RI(Create) Group',values='Nomor PO+RI').reset_index().merge(
                df_eksternal2[ 
                     (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Eksternal Resto')].groupby(['Rute'])[['Nomor PO+RI']].nunique().reset_index().rename(columns={'Nomor PO+RI':'Total'})
                     ),
                         hide_index=True)
        
        st.write('PIC Responsible: WH/CK')
        st.write('Kategori Item: Internal Logistic')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal2[ 
                     (df_eksternal2['Kategori Item']=='Internal Logistic')].groupby(['Kategori RI(Date)-RI(Create)'])[['Nomor PO+RI']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori RI(Date)-RI(Create)', values_column='Nomor PO+RI', title="OUTGOING BACKDATE",key='pie_2')
        with col[1]:
            df_line = df_eksternal2[(df_eksternal2['Kategori Item']=='Internal Logistic')
                     ].groupby(['Date PO'])[['Nomor PO+RI']].nunique().reset_index().rename(columns={'Nomor PO+RI':'Total'}).merge(df_eksternal2[ 
                      (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Internal Logistic')
                     ].groupby(['Date PO'])[['Nomor PO+RI']].nunique().reset_index(), how='left')
            df_line['Nomor PO+RI'] = (df_line['Nomor PO+RI']/df_line['Total'])*100
            create_line_chart(df_line, x_column='Date PO', y_column='Nomor PO+RI', title="DAILY BACKDATE",key='line_2')
        with col[2]:
            df_bar = df_eksternal2[
                     (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Internal Logistic')].groupby(['Rute'])[['Nomor PO+RI']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO+RI', key='bar_2')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal2[  
                     (df_eksternal2['Kategori Item']=='Internal Logistic')]['Nomor PO+RI'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-RI(Create)']=='On-Time']['Nomor PO+RI'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-RI(Create)']=='Backdate']['Nomor PO+RI'].values[0]), delta=None)
            
            st.dataframe(df_eksternal2[
                     (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Internal Logistic')].groupby(['Rute','RI(Date)-RI(Create) Group'])[['Nomor PO+RI']].nunique().reset_index().pivot(index='Rute',columns='RI(Date)-RI(Create) Group',values='Nomor PO+RI').reset_index().merge(
                df_eksternal2[ 
                     (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Internal Logistic')].groupby(['Rute'])[['Nomor PO+RI']].nunique().reset_index().rename(columns={'Nomor PO+RI':'Total'})
                     ),
                         hide_index=True)
            
    
    data = {
        "Tanggal": pd.date_range(start="2024-01-01", end="2025-12-31", freq="D"),
        "Penjualan": range(1, 732)
    }
    df = pd.DataFrame(data)
    
    
    # Widget untuk memilih rentang tanggal
    start_date, end_date = st.date_input(
        "RANGE DATE ",
        [df["Tanggal"].min(), df["Tanggal"].max()],  # Default nilai awal
        min_value=df["Tanggal"].min(),
        max_value=df["Tanggal"].max()
    )
    
    df_tanggal = pd.DataFrame(pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq='D'), columns=['Tanggal'])
    
    if kat_eksternal =='PR(Create)-PO(Datang)':
        st.markdown('### PR(Create)-PO(Datang)')
        st.write('PIC Responsible: Logistic')
        st.write('Kategori Item: Eksternal Logistic')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori Item']=='Eksternal Logistic')].groupby(['Kategori PR(Create)-PO(Datang)'])[['Nomor PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori PR(Create)-PO(Datang)', values_column='Nomor PO', title="OUTGOING BACKDATE")
        with col[1]:
            df_line = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Logistic')
                     ].groupby(['Tanggal PO'])[['Nomor PO']].nunique().reset_index()
            df_line = df_tanggal.merge(df_line,how='left',left_on='Tanggal',right_on='Tanggal PO').fillna(0)
            create_line_chart(df_line, x_column='Tanggal', y_column='Nomor PO', title="DAILY BACKDATE")
        with col[2]:
            df_bar = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Logistic')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori Item']=='Eksternal Logistic')]['Nomor PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori PR(Create)-PO(Datang)']=='On-Time']['Nomor PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori PR(Create)-PO(Datang)']=='Backdate']['Nomor PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Logistic')].groupby(['Rute','PR(Create)-PO(Datang) Group'])[['Nomor PO']].nunique().reset_index().pivot(index='Rute',columns='PR(Create)-PO(Datang) Group',values='Nomor PO').reset_index().merge(
                df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Logistic')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'})
                     ),
                         hide_index=True)
        
        st.write('Kategori Item: Internal Logistic')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori Item']=='Internal Logistic')].groupby(['Kategori PR(Create)-PO(Datang)'])[['Nomor PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori PR(Create)-PO(Datang)', values_column='Nomor PO', title="OUTGOING BACKDATE")
        with col[1]:
            df_line = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Internal Logistic')
                     ].groupby(['Tanggal PO'])[['Nomor PO']].nunique().reset_index()
            df_line = df_tanggal.merge(df_line,how='left',left_on='Tanggal',right_on='Tanggal PO').fillna(0)
            create_line_chart(df_line, x_column='Tanggal', y_column='Nomor PO', title="DAILY BACKDATE")
        with col[2]:
            df_bar = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Internal Logistic')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori Item']=='Internal Logistic')]['Nomor PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori PR(Create)-PO(Datang)']=='On-Time']['Nomor PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori PR(Create)-PO(Datang)']=='Backdate']['Nomor PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Internal Logistic')].groupby(['Rute','PR(Create)-PO(Datang) Group'])[['Nomor PO']].nunique().reset_index().pivot(index='Rute',columns='PR(Create)-PO(Datang) Group',values='Nomor PO').reset_index().merge(
                df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Internal Logistic')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'})
                     ),
                         hide_index=True)
            
        st.write('Kategori Item: Resto')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     ].groupby(['Kategori PR(Create)-PO(Datang)'])[['Nomor PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori PR(Create)-PO(Datang)', values_column='Nomor PO', title="OUTGOING BACKDATE")
        with col[1]:
            df_line = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate')
                     ].groupby(['Tanggal PO'])[['Nomor PO']].nunique().reset_index()
            df_line = df_tanggal.merge(df_line,how='left',left_on='Tanggal',right_on='Tanggal PO').fillna(0)
            create_line_chart(df_line, x_column='Tanggal', y_column='Nomor PO', title="DAILY BACKDATE")
        with col[2]:
            df_bar = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') ]['Nomor PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori PR(Create)-PO(Datang)']=='On-Time']['Nomor PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori PR(Create)-PO(Datang)']=='Backdate']['Nomor PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate')].groupby(['Rute','PR(Create)-PO(Datang) Group'])[['Nomor PO']].nunique().reset_index().pivot(index='Rute',columns='PR(Create)-PO(Datang) Group',values='Nomor PO').reset_index().merge(
                df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori PR(Create)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'})
                     ),
                         hide_index=True)
    
    if kat_eksternal =='PO(Create)-PO(Datang)':
        st.markdown('### Kategori PO(Create)-PO(Datang)')
        st.write('PIC Responsible: Procurement')
        st.write('Kategori Item: Logistic')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     ].groupby(['Kategori PO(Create)-PO(Datang)'])[['Nomor PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori PO(Create)-PO(Datang)', values_column='Nomor PO', title="OUTGOING BACKDATE")
        with col[1]:
            df_line = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate')
                     ].groupby(['Tanggal PO'])[['Nomor PO']].nunique().reset_index()
            df_line = df_tanggal.merge(df_line,how='left',left_on='Tanggal',right_on='Tanggal PO').fillna(0)
            create_line_chart(df_line, x_column='Tanggal', y_column='Nomor PO', title="DAILY BACKDATE")
        with col[2]:
            df_bar = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                    ]['Nomor PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori PO(Create)-PO(Datang)']=='On-Time']['Nomor PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori PO(Create)-PO(Datang)']=='Backdate']['Nomor PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate')].groupby(['Rute','PO(Create)-PO(Datang) Group'])[['Nomor PO']].nunique().reset_index().pivot(index='Rute',columns='PO(Create)-PO(Datang) Group',values='Nomor PO').reset_index().merge(
                df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'})
                     ),
                         hide_index=True)
            
        st.write('Kategori Item: Resto')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori Item']=='Eksternal Resto')].groupby(['Kategori PO(Create)-PO(Datang)'])[['Nomor PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori PO(Create)-PO(Datang)', values_column='Nomor PO', title="OUTGOING BACKDATE")
        with col[1]:
            df_line = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Resto')
                     ].groupby(['Tanggal PO'])[['Nomor PO']].nunique().reset_index()
            df_line = df_tanggal.merge(df_line,how='left',left_on='Tanggal',right_on='Tanggal PO').fillna(0)
            create_line_chart(df_line, x_column='Tanggal', y_column='Nomor PO', title="DAILY BACKDATE")
        with col[2]:
            df_bar = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Resto')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                    ]['Nomor PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori PO(Create)-PO(Datang)']=='On-Time']['Nomor PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori PO(Create)-PO(Datang)']=='Backdate']['Nomor PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate')].groupby(['Rute','PO(Create)-PO(Datang) Group'])[['Nomor PO']].nunique().reset_index().pivot(index='Rute',columns='PO(Create)-PO(Datang) Group',values='Nomor PO').reset_index().merge(
                df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori PO(Create)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'})
                     ),
                         hide_index=True)

    if kat_eksternal =='RI(Date)-PO(Datang)':
        st.markdown('### Kategori RI(Date)-PO(Datang)')
        st.write('PIC Responsible: Procurement')
        st.write('Kategori Item: Logistic')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     ].groupby(['Kategori RI(Date)-PO(Datang)'])[['Nomor PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori RI(Date)-PO(Datang)', values_column='Nomor PO', title="OUTGOING BACKDATE")
        with col[1]:
            df_line = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate')
                     ].groupby(['Tanggal PO'])[['Nomor PO']].nunique().reset_index()
            df_line = df_tanggal.merge(df_line,how='left',left_on='Tanggal',right_on='Tanggal PO').fillna(0)
            create_line_chart(df_line, x_column='Tanggal', y_column='Nomor PO', title="DAILY BACKDATE")
        with col[2]:
            df_bar = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                    ]['Nomor PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-PO(Datang)']=='On-Time']['Nomor PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-PO(Datang)']=='Backdate']['Nomor PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate')].groupby(['Rute','RI(Date)-PO(Datang) Group'])[['Nomor PO']].nunique().reset_index().pivot(index='Rute',columns='RI(Date)-PO(Datang) Group',values='Nomor PO').reset_index().merge(
                df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Logistic') 
                     & (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'})
                     ),
                         hide_index=True)

        st.write('Kategori Item: Resto')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori Item']=='Eksternal Resto')].groupby(['Kategori RI(Date)-PO(Datang)'])[['Nomor PO']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori RI(Date)-PO(Datang)', values_column='Nomor PO', title="OUTGOING BACKDATE")
        with col[1]:
            df_line = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Resto')
                     ].groupby(['Tanggal PO'])[['Nomor PO']].nunique().reset_index()
            df_line = df_tanggal.merge(df_line,how='left',left_on='Tanggal',right_on='Tanggal PO').fillna(0)
            create_line_chart(df_line, x_column='Tanggal', y_column='Nomor PO', title="DAILY BACKDATE")
        with col[2]:
            df_bar = df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate') & (df_eksternal['Kategori Item']=='Eksternal Resto')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                    ]['Nomor PO'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-PO(Datang)']=='On-Time']['Nomor PO'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-PO(Datang)']=='Backdate']['Nomor PO'].values[0]), delta=None)
            
            st.dataframe(df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate')].groupby(['Rute','RI(Date)-PO(Datang) Group'])[['Nomor PO']].nunique().reset_index().pivot(index='Rute',columns='RI(Date)-PO(Datang) Group',values='Nomor PO').reset_index().merge(
                df_eksternal[(df_eksternal["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal['PIC Responsible']=='Resto') 
                     & (df_eksternal['Kategori RI(Date)-PO(Datang)']=='Backdate')].groupby(['Rute'])[['Nomor PO']].nunique().reset_index().rename(columns={'Nomor PO':'Total'})
                     ),
                         hide_index=True)

    if kat_eksternal =='RI(Date)-RI(Create)':
        st.markdown('### Kategori RI(Date)-RI(Create)')
        st.write('PIC Responsible: Procurement')
        st.write('Kategori Item: Logistic')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal2[(df_eksternal2["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal2["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal2['PIC Responsible']=='Logistic') 
                     ].groupby(['Kategori RI(Date)-RI(Create)'])[['Nomor PO+RI']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori RI(Date)-RI(Create)', values_column='Nomor PO+RI', title="OUTGOING BACKDATE")
        with col[1]:
            df_line = df_eksternal2[(df_eksternal2["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal2["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal2['PIC Responsible']=='Logistic') 
                     & (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate')
                     ].groupby(['Tanggal PO'])[['Nomor PO+RI']].nunique().reset_index()
            df_line = df_tanggal.merge(df_line,how='left',left_on='Tanggal',right_on='Tanggal PO').fillna(0)
            create_line_chart(df_line, x_column='Tanggal', y_column='Nomor PO+RI', title="DAILY BACKDATE")
        with col[2]:
            df_bar = df_eksternal2[(df_eksternal2["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal2["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal2['PIC Responsible']=='Logistic') 
                     & (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate')].groupby(['Rute'])[['Nomor PO+RI']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO+RI')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal2[(df_eksternal2["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal2["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal2['PIC Responsible']=='Logistic') 
                    ]['Nomor PO+RI'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-RI(Create)']=='On-Time']['Nomor PO+RI'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-RI(Create)']=='Backdate']['Nomor PO+RI'].values[0]), delta=None)
            
            st.dataframe(df_eksternal2[(df_eksternal2["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal2["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal2['PIC Responsible']=='Logistic') 
                     & (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate')].groupby(['Rute','RI(Date)-RI(Create) Group'])[['Nomor PO+RI']].nunique().reset_index().pivot(index='Rute',columns='RI(Date)-RI(Create) Group',values='Nomor PO+RI').reset_index().merge(
                df_eksternal2[(df_eksternal2["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal2["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal2['PIC Responsible']=='Logistic') 
                     & (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate')].groupby(['Rute'])[['Nomor PO+RI']].nunique().reset_index().rename(columns={'Nomor PO+RI':'Total'})
                     ),
                         hide_index=True)
            
        st.write('Kategori Item: Resto')
        col = st.columns([1,2,1,2])
        
        with col[0]:
            df_pie = df_eksternal2[(df_eksternal2["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal2["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal2['PIC Responsible']=='Resto') 
                     & (df_eksternal2['Kategori Item']=='Eksternal Resto')].groupby(['Kategori RI(Date)-RI(Create)'])[['Nomor PO+RI']].nunique().reset_index()
            create_pie_chart(df_pie, labels_column='Kategori RI(Date)-RI(Create)', values_column='Nomor PO+RI', title="OUTGOING BACKDATE")
        with col[1]:
            df_line = df_eksternal2[(df_eksternal2["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal2["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal2['PIC Responsible']=='Resto') 
                     & (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Eksternal Resto')
                     ].groupby(['Tanggal PO'])[['Nomor PO+RI']].nunique().reset_index()
            df_line = df_tanggal.merge(df_line,how='left',left_on='Tanggal',right_on='Tanggal PO').fillna(0)
            create_line_chart(df_line, x_column='Tanggal', y_column='Nomor PO+RI', title="DAILY BACKDATE")
        with col[2]:
            df_bar = df_eksternal2[(df_eksternal2["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal2["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal2['PIC Responsible']=='Resto') 
                     & (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate') & (df_eksternal2['Kategori Item']=='Eksternal Resto')].groupby(['Rute'])[['Nomor PO+RI']].nunique().reset_index()
            create_percentage_barchart(df_bar, 'Rute', 'Nomor PO+RI')
        with col[3]:
            st.write('')
            col2 = st.columns(3)
            with col2[0]:
                st.metric(label="Total", value="{:,.0f}".format(df_eksternal2[(df_eksternal2["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal2["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal2['PIC Responsible']=='Resto') 
                    ]['Nomor PO+RI'].nunique()), delta=None)
            with col2[1]:
                st.metric(label="On-Time", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-RI(Create)']=='On-Time']['Nomor PO+RI'].values[0]), delta=None)
            with col2[2]:
                st.metric(label="Backdate", value="{:,.0f}".format(df_pie[df_pie['Kategori RI(Date)-RI(Create)']=='Backdate']['Nomor PO+RI'].values[0]), delta=None)
            
            st.dataframe(df_eksternal2[(df_eksternal2["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal2["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal2['PIC Responsible']=='Resto') 
                     & (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate')].groupby(['Rute','RI(Date)-RI(Create) Group'])[['Nomor PO+RI']].nunique().reset_index().pivot(index='Rute',columns='RI(Date)-RI(Create) Group',values='Nomor PO+RI').reset_index().merge(
                df_eksternal2[(df_eksternal2["Tanggal PO"] >= pd.Timestamp(start_date)) & (df_eksternal2["Tanggal PO"] <= pd.Timestamp(end_date)) & (df_eksternal2['PIC Responsible']=='Resto') 
                     & (df_eksternal2['Kategori RI(Date)-RI(Create)']=='Backdate')].groupby(['Rute'])[['Nomor PO+RI']].nunique().reset_index().rename(columns={'Nomor PO+RI':'Total'})
                     ),
                         hide_index=True)
