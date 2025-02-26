import streamlit as st
import requests
import zipfile
import io
import pandas as pd
import numpy as np
import os
import gdown
import tempfile



import plotly.express as px
import plotly.graph_objs as go


st.set_page_config(layout="wide")
def highlight_header(x):
    """
    Meng-highlight header tabel dengan warna merah.
    
    Parameters:
    - x: DataFrame yang akan diterapkan styling
    
    Returns:
    - DataFrame dengan styling untuk header
    """
    # CSS styling untuk header
    header_color = 'background-color: #FF4B4B; color: white;'  # Warna merah dengan teks putih
    df_styles = pd.DataFrame('', index=x.index, columns=x.columns)
    
    # Memberikan warna khusus pada header
    df_styles.loc[x.columns, :] = header_color

    return df_styles
    
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Fungsi untuk mereset state button
def reset_button_state():
    st.session_state.button_clicked = False

def download_file_from_google_drive(file_id, dest_path):
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)
def highlight_header(s):
    return ['background-color: red; color: white'] * len(s)
    

def load_excel(file_path):
    with open(file_path, 'rb') as file:
        model = pd.read_excel(file, engine='openpyxl')
    return model


col = st.columns(2)
with col[0]:
    st.title('Dashboard - Inventaris Control')
with col[1]:
    tahun = st.selectbox("TAHUN:", ['2024','2025'], index=1, on_change=reset_button_state)
        
if 'df_4101' not in locals():
    df_4101 = pd.read_csv(f'Dataset/Inventory Stock Monitoring/{tahun}/4101.csv')
    df_go = pd.read_excel('Dataset/Master/Resto GO.xlsx')
    df_db = pd.read_excel('Dataset/Master/Stocklevel.xlsx')
df_db=df_db.iloc[141:,0]

st.header('Total Biaya Inventaris all Cabang', divider='gray')

filter_akun = st.multiselect("Akun Penyesuaian:", ['Sparepart Inventaris - Resto (Non Jasa)', 'Sparepart Inventaris - DC (Non Jasa)', 'Sparepart Inventaris - CP (Non Jasa)'], default = ['Sparepart Inventaris - Resto (Non Jasa)', 'Sparepart Inventaris - DC (Non Jasa)', 'Sparepart Inventaris - CP (Non Jasa)'])

#df_4101 = df_4101[~(df_4101['Kode Barang'].astype(str).str.startswith('1')) & (df_4101['Akun Penyesuaian Persediaan']=='Sparepart Inventaris - Resto (Non Jasa)') & (df_4101['Nama Cabang'].str.startswith('1')) &
#        (df_4101['Kategori'].isin(['00.COST', '21.COST.ASSET', '20.ASSET.ASSET']))]
df_4101 =  df_4101[(df_4101['Akun Penyesuaian Persediaan'].isin(filter_akun)) & (df_4101['Nama Cabang'].str.startswith('1'))]
df_4101['Nama Barang'] = df_4101['Nama Barang'].str.replace('TEMPAT SAMPAH 80L (THAWING ADONAN)','BAK THAWING BULAT 80 L')
df_4101['Tanggal'] = pd.to_datetime(df_4101['Tanggal'], format="%d/%m/%Y")
df_4101['Month'] = df_4101['Tanggal'].dt.strftime('%B %Y')
df_4101['Month'] = pd.Categorical(df_4101['Month'], categories=df_4101.sort_values('Tanggal')['Month'].unique(), ordered=True)
#df_4101['Nama Cabang'] = df_4101['Nama Cabang'].replace({'1029.CBNTEN (NON-AKTIF)':'1187.SBRTUP','1029.CBNTEN':'1187.SBRTUP','1010.MLGKEN':'1209.MLGSOE'})
df_4101['Nama Cabang'] = df_4101['Nama Cabang'].replace({'1029.CBNTEN (NON-AKTIF)':'1029.CBNTEN'})

df_4101_1 = df_4101[(df_4101['Tipe Penyesuaian']=='Pengurangan')].groupby(['Month','Nama Cabang'])[[f'Total Biaya']].sum().reset_index()
df_4101_1['Month'] = pd.Categorical(df_4101_1['Month'], categories=df_4101.sort_values('Tanggal')['Month'].unique(), ordered=True)

pivot1= df_4101_1[df_4101_1['Month']!='December 2023'].pivot(index='Nama Cabang', columns='Month',values='Total Biaya').reset_index()
total = pd.DataFrame((pivot1.iloc[:,1:].sum(axis=0).values).reshape(1,len(pivot1.columns)-1),columns=pivot1.columns[1:])
total['Nama Cabang']='Total'+(pivot1['Nama Cabang'].str.len().max()+8)*' '
st.dataframe(pivot1, use_container_width=True, hide_index=True)
#st.dataframe(total.loc[:,[total.columns[-1]]+total.columns[:-1].to_list()], use_container_width=True, hide_index=True)

df_go['Month'] = pd.to_datetime(df_go['Tanggal']).dt.strftime('%B %Y')
df_new = df_go.drop(columns='Tanggal').merge(df_4101_1,how='inner')
df_new = df_new[['Nama Cabang','Total Biaya','Month']]
df_new = df_new[df_new['Month']!='December 2023']

total2 = df_new.groupby('Month')[['Total Biaya']].sum().rename(columns={'Total Biaya':'Total [Cabang GO]'}).reset_index().merge(total.T.reset_index(),how='left').rename(columns={0:'%'})
total2['%'] = total2['Total [Cabang GO]']/total2['%']
total2 = total2.T
total2.columns=total2.iloc[0,]
total2 = total2.iloc[1:,:].rename(columns={'Month':'Nama Cabang'})

st.dataframe(pd.concat([total,
                        total2.reset_index().iloc[[0],:].rename(columns={'index':'Nama Cabang'})]).loc[:,[total.columns[-1]]+total.columns[:-1].to_list()], use_container_width=True, hide_index=True)

dfs = []
k =' '
for bulan in df_4101_1[df_4101_1['Month']!='December 2023']['Month'].unique():
    df_new1 =df_new[df_new['Month']==bulan]
    df_new1.columns = [df_new1['Month'].unique()[0],k,'']
    k+=' '
    dfs.append(df_new1.iloc[:,:-1].reset_index(drop=True))

df_new = pd.concat(dfs,axis=1,ignore_index=True)
i=0
y=' '
for i, x in enumerate(df_4101_1[df_4101_1['Month']!='December 2023']['Month'].unique()):
    df_new = df_new.rename(columns={i*2:x})
    df_new = df_new.rename(columns={1+(i*2):y})
    y+=' '
    i+=1
    
st.header('Total Biaya Inventaris Cabang GO', divider='gray')
st.dataframe(df_new, use_container_width=True, hide_index=True)

y = 0
i = [] 
kol = []
for x in df_new.columns:
    if y%2!=0:
        i.append(x)
    else:
        kol.append(x)
    y+=1


st.dataframe(pd.DataFrame([df_new.loc[:,i].sum().values],columns=kol), use_container_width=True, hide_index=True)

st.header('Detail', divider='gray')
col = st.columns(4)
with col[0]:
    cabang = st.selectbox("NAMA CABANG:", ['All'] + sorted(df_4101['Nama Cabang'].unique().tolist()), index=0, on_change=reset_button_state)
with col[1]:
    tipe = st.selectbox("PENAMBAHAN/PENGURANGAN:", ['Penambahan','Pengurangan'], index=1, on_change=reset_button_state)
with col[2]:
    qty_nom = st.selectbox("KUANTITAS/TOTAL BIAYA:", ['Kuantitas','Total Biaya'], index=0, on_change=reset_button_state)
with col[3]:
    kategori = st.selectbox("KATEGORI ITEM:", ['All','Sparepart Inventaris - Resto (Non Jasa)','Maintenance and Repair'], index=0, on_change=reset_button_state)

list_bulan = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December']

if kategori != 'All':
    if kategori == 'Maintenance and Repair':
        df_4101 = df_4101[(df_4101['Nama Barang'].isin(df_db.values.reshape(-1)))]
    else:
        df_4101 = df_4101[~(df_4101['Nama Barang'].isin(df_db.values.reshape(-1)))]
        
if cabang != 'All':
    df_4101 = df_4101[(df_4101['Nama Cabang']==cabang)]

#df_4101 = df_4101[df_4101['Month']>='January 2024']
df_4101 = df_4101[(df_4101['Tipe Penyesuaian']== tipe)]
df_4101_1 = df_4101.groupby(['Month','Nama Barang'])[[f'{qty_nom}']].sum().reset_index()

df_4101_1['Month'] = pd.Categorical(df_4101_1['Month'], categories=df_4101.sort_values('Tanggal')['Month'].unique(), ordered=True)
df_4101_1 = df_4101_1.sort_values('Month')
month = df_4101_1['Month'].unique().tolist()
df_4101_1 = df_4101_1.pivot(index='Nama Barang', columns='Month',values=f'{qty_nom}').reset_index().fillna(0)
#df_4101_1.iloc[:,1:] = df_4101_1.iloc[:,1:].applymap(lambda x: '' if x=='' else f'{x:.0f}')
df_4101_1.iloc[:,1:] = df_4101_1.iloc[:,1:]#.astype(int)
total = pd.DataFrame((df_4101_1.iloc[:,1:].sum(axis=0).values).reshape(1,len(df_4101_1.columns)-1),columns=df_4101_1.columns[1:])
total['Nama Barang']='TOTAL'+(df_4101_1['Nama Barang'].str.len().max()+8)*' '

#pd.options.st.dataframe.float_format = '{:,.0f}'.format
df_4101_2 = df_4101.groupby(['Nama Cabang','Nomor #','Kode Barang','Nama Barang','Tipe Penyesuaian'])[['Kuantitas','Total Biaya']].sum().reset_index()
df_4101_2 = df_4101_2.pivot(index=['Nama Cabang','Nomor #','Kode Barang','Nama Barang'],columns=['Tipe Penyesuaian'],values=['Kuantitas','Total Biaya']).reset_index().fillna(0)

def highlight_header(s):
    return ['background-color: red; color: white;' for _ in s]

# Mengaplikasikan style ke DataFrame
st.dataframe(pd.concat([df_4101_1,total])[:-1], use_container_width=True, hide_index=True)
st.dataframe(pd.concat([df_4101_1,total])[-1:], use_container_width=True, hide_index=True)

if cabang != 'All':
    all_month = []
    for i in month:
        all_month.append(pd.DataFrame(df_4101[df_4101['Month']==f'{i}']['Nomor #'].unique(),columns=[f'{i}']))
    df_ia = pd.concat(all_month,axis=1, ignore_index=True)
    for i, x in enumerate(month):
        df_ia = df_ia.rename(columns={i:x})
    df_ia['Nama Cabang'] = cabang
    df_ia = df_ia[[df_ia.columns[-1]]+list(df_ia.columns[:-1])].fillna('')
    
    st.markdown('### Daftar Nomor IA')
    st.dataframe(df_ia, use_container_width=True, hide_index=True)
    
    st.markdown('### Detail Nomor IA')
    list_ia = sorted(df_4101_2['Nomor #'].unique().tolist())
    ia = st.selectbox("NOMOR IA:",list_ia ,index=len(list_ia)-1, on_change=reset_button_state)
    df_4101_2 = df_4101_2[df_4101_2['Nomor #'] == ia].drop(columns='Nomor #')
    df_4101_2.columns = ['_'.join(col).strip() for col in df_4101_2.columns.values]
    
    
    total = pd.DataFrame((df_4101_2.iloc[:,3:].sum(axis=0).values).reshape(1,len(df_4101_2.columns)-3),columns=df_4101_2.columns[3:])
    total['Nama Barang_']='TOTAL'+(df_4101_2['Nama Barang_'].str.len().max()+8)*' '
    #df_4101_2.iloc[:,3:] = df_4101_2.iloc[:,3:].astype(int)

    #df_4101_2.iloc[:,3:] = df_4101_2.iloc[:,3:].applymap(lambda x: '' if x=='' else f'{x:,.0f}')
    st.dataframe(pd.concat([df_4101_2,total])[:-1], use_container_width=True, hide_index=True)
    st.dataframe(pd.concat([df_4101_2,total])[-1:], use_container_width=True, hide_index=True)
