import streamlit as st
import requests
import zipfile
import io
import pandas as pd
import os
import gdown
import tempfile
import matplotlib.pyplot as plt
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


def download_file_from_google_drive(file_id, dest_path):
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)

list_cab = pd.read_excel('Dataset/Master/list_cab.xlsx')

# Unduh file dari GitHub


if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Fungsi untuk mereset state button
def reset_button_state():
    st.session_state.button_clicked = False
    
col = st.columns(2)
with col[0]:
    st.title('Dashboard - Selisih Ojol')
with col[1]:
    tahun = st.selectbox("TAHUN:", ['2024','2025'], index=1, on_change=reset_button_state)

if 'df_merge' not in locals():
    df_selisih = pd.read_csv(f'Dataset/Selisih Ojol/{tahun}/df_selisih.csv')
    df_merge = pd.read_csv(f'Dataset/Selisih Ojol/{tahun}/merge_clean.csv')
    df_breakdown = pd.read_csv(f'Dataset/Selisih Ojol/{tahun}/breakdown_clean.csv')
    pic = pd.read_excel(f'Dataset/Selisih Ojol/{tahun}/PIC Ojol.xlsx')
    f=f'Dataset/Selisih Ojol/{tahun}/CNS_NASIONAL.xlsx'
    s_nas = pd.read_excel(f,sheet_name='SELISIH')
    cn_nas = pd.read_excel(f,sheet_name='CANCELNOTA')
    oms_nas = pd.read_excel(f,sheet_name='OMSET')

df_selisih['MONTH'] = pd.to_datetime(df_selisih['MONTH'])
df_merge['MONTH'] = pd.to_datetime(df_merge['MONTH'])
df_breakdown['MONTH'] = pd.to_datetime(df_breakdown['MONTH'])

s_nas = s_nas[s_nas['MONTH']>'2024-01-01']
oms_nas = oms_nas[oms_nas['MONTH']>'2024-01-01']
cn_nas = cn_nas[cn_nas['MONTH']>'2024-01-01']
pic = pic[pic['BULAN']>'2024-01-01']
df_selisih = df_selisih[df_selisih['MONTH']>'2024-01-01']
df_merge = df_merge[df_merge['MONTH']>'2024-01-01']
df_breakdown = df_breakdown[df_breakdown['MONTH']>'2024-01-01']

df_month = s_nas[['MONTH']].copy()
df_month['BULAN'] = df_month['MONTH'].dt.strftime('%B')

s_nas['MONTH'] = s_nas['MONTH'].dt.strftime('%B')
oms_nas['MONTH'] = oms_nas['MONTH'].dt.strftime('%B')
cn_nas['MONTH'] = cn_nas['MONTH'].dt.strftime('%B')
pic['BULAN'] = pic['BULAN'].dt.strftime('%B')
df_selisih['MONTH'] = df_selisih['MONTH'].dt.strftime('%B')
df_merge['MONTH'] = df_merge['MONTH'].dt.strftime('%B')
df_breakdown['MONTH'] = df_breakdown['MONTH'].dt.strftime('%B')

s_nas['SELISIH'] = abs(s_nas['SELISIH'])
oms_nas['OMSET'] = abs(oms_nas['OMSET'])
cn_nas['CANCEL NOTA'] = abs(cn_nas['CANCEL NOTA'])

def highlight_last_row(x):
    font_color = 'color: white;'
    background_color = 'background-color: #FF4B4B;'  # Warna yang ingin digunakan
    df_styles = pd.DataFrame('', index=x.index, columns=x.columns)
    
    # Memberikan warna khusus pada baris terakhir yang bernama 'SELISIH'
    df_styles.iloc[-1, :] = font_color + background_color

    return df_styles
               
def format_number(x):
    if x==0:
        return ''
    if isinstance(x, (int, float)):
        return "{:,.0f}".format(x)
    return x

kat_pengurang = ['Invoice Beda Hari',
                 'Transaksi Kemarin',
                 'Selisih IT',
                 'Promo Marketing/Adjustment',
                 'Cancel Nota',
                 'Tidak Ada Transaksi di Web',
                 'Selisih Lebih Bayar QRIS',
                 'Selisih Lebih Bayar Ojol',
                 'Salah Slot Pembayaran']
kat_diperiksa = ['Tidak Ada Invoice QRIS',
                 'Tidak Ada Invoice Ojol',
                 'Double Input',
                 'Selisih Kurang Bayar QRIS',
                 'Selisih Kurang Bayar Ojol',
                 'Bayar Lebih dari 1 Kali - 1 Struk (QRIS)',
                 'Bayar 1 Kali - Banyak Struk (QRIS)',
                 'Bayar Lebih dari 1 Kali - Banyak Struk (QRIS)',
                 'Kurang Input (Ojol)']

df_pic_oms = oms_nas.rename(columns={'BULAN':'MONTH'}).groupby(['MONTH','CAB'])[['OMSET']].sum().reset_index()
df_pic_oms['MONTH'] = pd.Categorical(df_pic_oms['MONTH'], categories=df_month['BULAN'].unique(), ordered=True)
df_pic_oms = df_pic_oms.sort_values('MONTH')

pic['BULAN'] = pd.Categorical(pic['BULAN'], categories=df_month['BULAN'].unique(), ordered=True)
pic = pic.sort_values('BULAN')
pic2 = pic.groupby('NAMA RESTO')[['BULAN']].max().reset_index().merge(pic).drop(columns='BULAN')
df_pic_oms = df_pic_oms.merge(pic2,how='left',left_on=['CAB'],right_on =['NAMA RESTO']).groupby(['NAMA PIC','MONTH','CAB'])[['OMSET']].sum().reset_index()
df_pic_oms['OMSET'] = abs(df_pic_oms['OMSET'])
#df_pic_oms = pd.concat([df_pic_oms,df_pic_oms2],ignore_index=True)
df_pic_oms = df_pic_oms[df_pic_oms['OMSET']!=0]
df_pic_oms['MONTH'] = pd.Categorical(df_pic_oms['MONTH'], categories=df_month['BULAN'].unique(), ordered=True)
df_pic_oms = df_pic_oms.sort_values(['NAMA PIC','MONTH'])
df_pic_oms = df_pic_oms.pivot(index=['NAMA PIC','CAB'],columns='MONTH',values='OMSET').reset_index().reset_index()

df_pic = df_breakdown[df_breakdown['Kategori'].isin([x.upper() for x in kat_diperiksa])].groupby(['MONTH','CAB'])[df_breakdown.columns[-5:]].sum().sum(axis=1).reset_index().rename(columns={0:'SELISIH'})
df_pic['MONTH'] = pd.Categorical(df_pic['MONTH'], categories=df_month['BULAN'].unique(), ordered=True)
df_pic = df_pic.sort_values('MONTH')

df_pic = df_pic.merge(pic,how='left',left_on=['CAB','MONTH'],right_on =['NAMA RESTO','BULAN']).groupby(['NAMA PIC','MONTH','CAB'])[['SELISIH']].sum().reset_index()
df_pic['SELISIH'] = abs(df_pic['SELISIH'])
#df_pic = pd.concat([df_pic,df_pic2],ignore_index=True)
df_pic = df_pic[df_pic['SELISIH']!=0]
df_pic['MONTH'] = pd.Categorical(df_pic['MONTH'], categories=df_month['BULAN'].unique(), ordered=True)
df_pic = df_pic.sort_values(['NAMA PIC','MONTH'])
df_pic = df_pic.pivot(index=['NAMA PIC','CAB'],columns='MONTH',values='SELISIH').reset_index().reset_index()
df_pic = df_pic.melt(id_vars=['index','NAMA PIC','CAB'])

df_pic2 = df_pic[(df_pic['value'].isna())]
df_pic1 = df_pic[~(df_pic['value'].isna())].rename(columns={'value':'SELISIH'})
df_pic2 = df_pic2.merge(s_nas,how='left').fillna(0).drop(columns='value')

df_pic = pd.concat([df_pic1,df_pic2],ignore_index=True)
df_pic['MONTH'] = pd.Categorical(df_pic['MONTH'], categories=df_month['BULAN'].unique(), ordered=True)
df_pic = df_pic.sort_values(['NAMA PIC','MONTH']).pivot(index=['NAMA PIC','CAB'],columns='MONTH',values='SELISIH').reset_index()
#df_pic = df_pic.fillna(0).style.format(lambda x: format_number(x)).background_gradient(cmap='Reds', axis=1, subset=df_pic.columns[2:])

def highlight_cells(x, highlight_info=df_pic2.drop(columns=['CAB','NAMA PIC','SELISIH'])):
    # Membuat DataFrame kosong dengan warna default (tidak ada warna)
    df_styles = pd.DataFrame('', index=x.index, columns=x.columns)
    
    # Iterasi melalui highlight_info untuk mengisi DataFrame styles dengan warna
    for idx, row in highlight_info.iterrows():
        # Menentukan warna untuk sel yang dipilih
        row_index = row['index']
        col_name = row['MONTH']
        
        # Memeriksa apakah row_index dan col_name ada di DataFrame
        if row_index in df_styles.index and col_name in df_styles.columns:
            df_styles.at[row_index, col_name] = 'background-color: yellow;'
    
    return df_styles
    
styled_pivot_df = df_pic_oms.fillna(0).drop(columns='index').style.format(lambda x: format_number(x)).background_gradient(cmap='Reds', axis=1, subset=df_pic.columns[2:])
st.markdown('### OMSET')
st.dataframe(styled_pivot_df, use_container_width=True, hide_index=True) 
styled_pivot_df = df_pic.style.format(lambda x: format_number(x)).background_gradient(cmap='Reds', axis=1, subset=df_pic.columns[2:]).apply(highlight_cells, highlight_info=df_pic2.drop(columns=['CAB','NAMA PIC','SELISIH']), axis=None).set_properties(**{'color': 'black'})
st.markdown('### SELISIH')
st.dataframe(styled_pivot_df, use_container_width=True, hide_index=True) 


df_snas = df_pic1.groupby(['MONTH'])[['SELISIH']].sum().reset_index()
df_snas['SELISIH NASIONAL'] = 0

for b in df_snas['MONTH']:
    df_snas.loc[df_snas[df_snas['MONTH']==b].index,'SELISIH NASIONAL'] = s_nas[(s_nas['MONTH']==b)&~(s_nas['CAB'].isin(df_pic1[(df_pic1['MONTH']==b)]['CAB'].values))]['SELISIH'].sum()

df_snas['%_SELISIH'] =df_snas['SELISIH']/(df_snas['SELISIH'] + df_snas['SELISIH NASIONAL'])
df_snas['%_SELISIH NASIONAL'] = df_snas['SELISIH NASIONAL']/(df_snas['SELISIH'] + df_snas['SELISIH NASIONAL'])
df_snas['MONTH'] = pd.Categorical(df_snas['MONTH'], categories=df_month['BULAN'].unique(), ordered=True)
df_snas = df_snas.sort_values(['MONTH'])
st.markdown("#### SELISIH BREAKDOWN vs SELISIH NASIONAL")
create_stylish_line_plot(df_snas, 'MONTH', '%_SELISIH', '%_SELISIH NASIONAL', title="", x_label="Month", y_label="Percentage")

df_cn = df_breakdown[df_breakdown['Kategori']=='CANCEL NOTA'].groupby(['MONTH','CAB'])[df_breakdown.columns[-5:]].sum().sum(axis=1).reset_index().rename(columns={0:'CANCEL NOTA'})
df_cn['MONTH'] = pd.Categorical(df_cn['MONTH'], categories=df_month['BULAN'].unique(), ordered=True)
df_cn = df_cn.sort_values('MONTH')
df_cn['CANCEL NOTA'] = abs(df_cn['CANCEL NOTA'])
df_cn = df_breakdown.groupby(['MONTH','CAB'])[['Kategori']].count().reset_index().drop(columns='Kategori').merge(df_cn,how='left').fillna(0)

df_cnnas = df_cn.groupby(['MONTH'])[['CANCEL NOTA']].sum().reset_index()
df_cnnas['CANCEL NOTA NASIONAL'] = 0

for b in df_cnnas['MONTH']:
    df_cnnas.loc[df_cnnas[df_cnnas['MONTH']==b].index,'CANCEL NOTA NASIONAL'] = cn_nas[(cn_nas['MONTH']==b)&~(cn_nas['CAB'].isin(df_pic1[(df_pic1['MONTH']==b)]['CAB'].values))]['CANCEL NOTA'].sum()

df_cnnas['%_CANCEL NOTA'] =df_cnnas['CANCEL NOTA']/(df_cnnas['CANCEL NOTA'] + df_cnnas['CANCEL NOTA NASIONAL'])
df_cnnas['%_CANCEL NOTA NASIONAL'] = df_cnnas['CANCEL NOTA NASIONAL']/(df_cnnas['CANCEL NOTA'] + df_cnnas['CANCEL NOTA NASIONAL'])
df_cnnas['MONTH'] = pd.Categorical(df_cnnas['MONTH'], categories=df_month['BULAN'].unique(), ordered=True)
df_cnnas = df_cnnas.sort_values(['MONTH'])
st.markdown("#### CANCEL NOTA BREAKDOWN vs CANCEL NOTA NASIONAL")
create_stylish_line_plot(df_cnnas, 'MONTH', '%_CANCEL NOTA', '%_CANCEL NOTA NASIONAL', title="", x_label="Month", y_label="Percentage")

all_cab_selisih = st.multiselect('Pilih Cabang', list_cab['CAB'].sort_values().unique().tolist()+['All'],default=['All'])
all_cab_selisih = list(all_cab_selisih)

df_selisih = df_selisih[df_selisih['CAB'].isin(df_selisih['CAB'].unique() if 'All' in all_cab_selisih else all_cab_selisih)]
df_selisih['MONTH'] = pd.Categorical(df_selisih['MONTH'], categories=df_month['BULAN'].unique(), ordered=True)
df_selisih = df_selisih.sort_values('MONTH')
df_selisih['%_CANCEL NOTA'] = df_selisih['CANCEL NOTA']/df_selisih['TOTAL']
df_selisih['%_DOUBLE INPUT'] = df_selisih['DOUBLE INPUT']/df_selisih['TOTAL']
df_selisih['%_TIDAK ADA INVOICE OJOL'] = df_selisih['TIDAK ADA INVOICE OJOL']/df_selisih['TOTAL']
df_selisih['%_TIDAK ADA INVOICE QRIS'] = df_selisih['TIDAK ADA INVOICE QRIS']/df_selisih['TOTAL']
df_selisih['%_SELISIH'] = df_selisih['SELISIH']/df_selisih['TOTAL']
st.markdown("#### SELISIH BREAKDOWN vs CANCEL NOTA BREAKDOWN")
metrik = st.radio('',['Avg','Total'], horizontal=True,label_visibility="collapsed")
if metrik =='Avg':
    df_selisih = df_selisih.groupby(['MONTH'])[df_selisih.columns[2:]].mean().reset_index()
else:
    df_selisih = df_selisih.groupby(['MONTH'])[df_selisih.columns[2:]].sum().reset_index()
df_selisih = df_selisih.dropna(axis=0,subset=df_selisih.columns[1:])
create_stylish_line_plot(df_selisih, 'MONTH', '%_SELISIH', '%_CANCEL NOTA', title="", x_label="Month", y_label="Percentage")
df_selisih2 = pd.DataFrame(df_selisih.iloc[:,:-7].T.reset_index().values[1:], columns=df_selisih.iloc[:,:-7].T.reset_index().values[0]).dropna(axis=1, how='all').applymap(format_number)
st.dataframe(df_selisih2, use_container_width=True, hide_index=True)
    
st.title('Data - Selisih Ojol')
col = st.columns(2)

with col[0]:
    all_cab = st.multiselect('Pilih Cabang', list_cab['CAB'].sort_values().unique(), on_change=reset_button_state)
    all_cab = list(all_cab)

with col[1]:
    list_bulan = df_month['BULAN'].unique().tolist()
    all_bulan = st.multiselect('Pilih Bulan', list_bulan, on_change=reset_button_state)
    

            
# Tombol untuk mengeksekusi aksi
if st.button('Show'):
    st.session_state.button_clicked = True
    
# Eksekusi kode jika tombol diklik
if st.session_state.button_clicked:
        df_merge = df_merge[df_merge['MONTH'].isin(all_bulan)]
        df_breakdown = df_breakdown[df_breakdown['MONTH'].isin(all_bulan)]

        for cab in all_cab:
            df_merge2 = df_merge[(df_merge['CAB'] == cab)]
            df_merge2 = df_merge2.groupby(['MONTH','SOURCE','KAT'])[['NOM']].sum().reset_index()
            for bulan in all_bulan:
                for i in ['GO RESTO','GRAB FOOD','QRIS SHOPEE','SHOPEEPAY']:
                    if i not in df_merge2[df_merge2['MONTH']==bulan]['KAT'].values:
                        df_merge2.loc[len(df_merge2)] = [bulan,'INVOICE',i,0]
                        df_merge2.loc[len(df_merge2)] = [bulan,'WEB',i,0]
            df_merge3 = df_merge2[df_merge2['KAT'].isin(['QRIS ESB','QRIS TELKOM'])].groupby(['MONTH','SOURCE'])[['NOM']].sum().reset_index()
            df_merge3['KAT']='QRIS TELKOM/ESB'
            
            for bulan in all_bulan:
                if df_merge3[df_merge3['MONTH']==bulan].empty:
                    df_merge3.loc[len(df_merge3)] = [bulan,'INVOICE',0,'QRIS TELKOM/ESB']
                    df_merge3.loc[len(df_merge3)] = [bulan, 'WEB',0,'QRIS TELKOM/ESB']
            df_merge_final = pd.concat([df_merge2[df_merge2['KAT'].isin(['GO RESTO','GRAB FOOD','QRIS SHOPEE','SHOPEEPAY'])],df_merge3]).sort_values('MONTH')
 
            
                
            st.markdown(f'## {cab}')
            st.markdown('#### SELISIH PER-PAYMENT')


            
            col = st.columns(len(all_bulan))
            for i, bulan in enumerate(all_bulan):
                with col[i]:
                    st.write(f'{bulan}')
                    df_merge_bln = pd.pivot(data=df_merge_final[df_merge_final['MONTH']==bulan], 
                                index='SOURCE', columns=['KAT'], values='NOM').reset_index().fillna(0)
                    df_merge_bln.loc[len(df_merge_bln)] =['SELISIH']+list(df_merge_bln.iloc[0,].values[1:] - df_merge_bln.iloc[1,].values[1:])
                    # Terapkan format pada seluruh DataFrame
                    df_merge_bln = df_merge_bln.applymap(format_number)
                    # Menerapkan styling pada DataFrame
                    df_merge_bln = df_merge_bln.style.apply(highlight_last_row, axis=None)

                    # Menampilkan DataFrame di Streamlit
                    st.dataframe(df_merge_bln, use_container_width=True, hide_index=True)
           
                    
            st.markdown('#### KATEGORI PENGURANG')
            df_breakdown2 = df_breakdown[df_breakdown['CAB'] == cab]
            df_breakdown_pengurang = df_breakdown2[df_breakdown2['Kategori'].isin([x.upper() for x in kat_pengurang])].groupby(['MONTH','Kategori'])[df_breakdown.columns[-5:]].sum().reset_index()
            col = st.columns(len(all_bulan))
            for i, bulan in enumerate(all_bulan):
                with col[i]:
                    st.write(f'{bulan}')
                    df_breakdown_pengurang_bln = df_breakdown_pengurang[df_breakdown_pengurang['MONTH']==bulan].iloc[:,1:].reset_index(drop=True)
                    df_breakdown_pengurang_bln.loc[len(df_breakdown_pengurang_bln)] = ['TOTAL',
                                                                              df_breakdown_pengurang_bln.iloc[:,1].sum(),
                                                                              df_breakdown_pengurang_bln.iloc[:,2].sum(),
                                                                              df_breakdown_pengurang_bln.iloc[:,3].sum(),
                                                                              df_breakdown_pengurang_bln.iloc[:,4].sum(),
                                                                              df_breakdown_pengurang_bln.iloc[:,5].sum()]
                    df_breakdown_pengurang_bln = df_breakdown_pengurang_bln.applymap(format_number)
                    df_breakdown_pengurang_bln = df_breakdown_pengurang_bln.style.apply(highlight_last_row, axis=None)
                    st.dataframe(df_breakdown_pengurang_bln, use_container_width=True, hide_index=True)
    
            st.markdown('#### KATEGORI DIPERIKSA')
            df_breakdown_diperiksa = df_breakdown2[df_breakdown2['Kategori'].isin([x.upper() for x in kat_diperiksa])].groupby(['MONTH','Kategori'])[df_breakdown.columns[-5:]].sum().reset_index()
            col = st.columns(len(all_bulan))
            for i, bulan in enumerate(all_bulan):
                with col[i]:
                    st.write(f'{bulan}')
                    df_breakdown_diperiksa_bln = df_breakdown_diperiksa[df_breakdown_diperiksa['MONTH']==bulan].iloc[:,1:].reset_index(drop=True)
                    df_breakdown_diperiksa_bln.loc[len(df_breakdown_diperiksa_bln)] = ['TOTAL',
                                                                              df_breakdown_diperiksa_bln.iloc[:,1].sum(),
                                                                              df_breakdown_diperiksa_bln.iloc[:,2].sum(),
                                                                              df_breakdown_diperiksa_bln.iloc[:,3].sum(),
                                                                              df_breakdown_diperiksa_bln.iloc[:,4].sum(),
                                                                              df_breakdown_diperiksa_bln.iloc[:,5].sum()]
                    df_breakdown_diperiksa_bln = df_breakdown_diperiksa_bln.applymap(format_number)
                    df_breakdown_diperiksa_bln = df_breakdown_diperiksa_bln.style.apply(highlight_last_row, axis=None)
                    st.dataframe(df_breakdown_diperiksa_bln, use_container_width=True, hide_index=True)
                    
            st.markdown('---')
        df = None
        dfs = None
        df_merge = None
        df_merge_final = None
        df_breakdown = None
        df_breakdown2 = None
        df_breakdown_diperiksa = None
        df_breakdown_pengurang = None
        st.cache_data.clear()
        st.cache_resource.clear()
