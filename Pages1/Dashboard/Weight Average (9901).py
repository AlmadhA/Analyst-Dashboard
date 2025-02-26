import pandas as pd
import glob
import numpy as np
import time
import datetime as dt
import re
import streamlit as st
from io import BytesIO
from xlsxwriter import Workbook
import pytz
import requests
import os
import tempfile
import zipfile

def download_file_from_github(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved to {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

st.title('Weight Average (99.01)')
uploaded_file = st.file_uploader("Upload File", type="zip")

def get_current_time_gmt7():
    tz = pytz.timezone('Asia/Jakarta')
    return dt.datetime.now(tz).strftime('%Y%m%d_%H%M%S')

if uploaded_file is not None:
    #st.write('File berhasil diupload')
    # Baca konten zip file

    if st.button('Process'):
        with st.spinner('Data sedang diproses...'):
          with zipfile.ZipFile(uploaded_file, 'r') as z:
              concatenated_df= []
              for file_name in z.namelist():
                  with z.open(file_name) as f:
                      # Membaca file Excel ke dalam DataFrame
                      df =   pd.read_excel(f)
                      concatenated_df.append(df) 
                      
        df_9901 = pd.concat(concatenated_df, ignore_index=True)
        # Assuming df is your DataFrame and these are the columns you mentioned
        columns_to_clean = ['#Purch.Qty', '#Purch.@Price', '#Purch.Discount', '#Purch.Total', '#Prime.Ratio', '#Prime.Qty', '#Prime.NetPrice']
        
        # Remove commas from values in specified columns
        #for col in columns_to_clean:
        # df_9901[col] = df_9901[col].str.replace(',', '')
        #Udang Kupas - CP replace Udang Thawing
        df_9901['Kode #'] = df_9901['Kode #'].astype(str)
        df_9901['Kode #'] = df_9901['Kode #'].replace('100084', '100167')
        df_9901['Nama Barang'] = df_9901['Nama Barang'].replace('UDANG KUPAS - CP', 'UDANG THAWING')
        
        numeric_cols = ['#Purch.Qty', '#Prime.Ratio', '#Prime.Qty', '#Purch.@Price', '#Purch.Discount', '#Prime.NetPrice', '#Purch.Total']
        df_9901[numeric_cols] = df_9901[numeric_cols].apply(pd.to_numeric)

        #RESTO
        
        # Calculate the total purchase amount for each item in each branch
        df_9901['Total_Purchase'] = df_9901.groupby(['Pemasok', 'Kode #', 'Month'])['#Purch.Total'].transform('sum')
        
        # Calculate the total quantity of prime items sold for each item in each branch
        df_9901['Total_Quantity'] = df_9901.groupby(['Pemasok', 'Kode #', 'Month'])['#Prime.Qty'].transform('sum')
        
        # Calculate the weighted average purchase total for each item in each branch
        df_9901['Weighted_Average Resto2'] = df_9901['Total_Purchase'] / df_9901['Total_Quantity']
        
        # Buat kolom WA 2 dan inisialisasi dengan nilai dari WA 1
        df_9901['Weighted_Average Resto'] = df_9901['Weighted_Average Resto2']
        
        # Mengosongkan kolom WA 2 untuk baris duplikat berdasarkan Nama Cabang, Kode #, dan Month
        df_9901.loc[df_9901.duplicated(subset=['Pemasok', 'Kode #', 'Month'], keep='first'), 'Weighted_Average Resto'] = ''
        
        #NASIONAL
        
        # Calculate the total purchase amount for each item in each branch
        df_9901['Total_Purchase'] = df_9901.groupby(['Kode #', 'Month'])['#Purch.Total'].transform('sum')
        
        # Calculate the total quantity of prime items sold for each item in each branch
        df_9901['Total_Quantity'] = df_9901.groupby(['Kode #', 'Month'])['#Prime.Qty'].transform('sum')
        
        # Calculate the weighted average purchase total for each item in each branch
        df_9901['Weighted_Average2'] = df_9901['Total_Purchase'] / df_9901['Total_Quantity']
        
        # Buat kolom WA 2 dan inisialisasi dengan nilai dari WA 1
        df_9901['Weighted_Average'] = df_9901['Weighted_Average2']
        
        # Mengosongkan kolom WA 2 untuk baris duplikat berdasarkan Nama Cabang, Kode #, dan Month
        df_9901.loc[df_9901.duplicated(subset=['Month', 'Kode #'], keep='first'), 'Weighted_Average'] = ''
        
        
        df_9901 = df_9901.drop(columns=['Total_Purchase','Total_Quantity'])
        
        df_pic  =   pd.read_csv('Dataset/Master/PIC v.2.csv').drop(columns=['Nama Barang','Kategori Barang'])
        df_pic['Kode #'] = df_pic['Kode #'].astype('int64')
        df_9901['Kode #'] = df_9901['Kode #'].astype('int64')
        df_9901 = pd.merge(df_9901, df_pic, how='left', on='Kode #').fillna('')
        
        df_9901 = df_9901[df_9901['Kategori Barang'].isin(['10.FOOD [RM] - COM', '10.FOOD [WIP] - COM', '01. COST - PACKAGING'])]
        cekblank    =   df_9901[df_9901['PIC'].isnull() | (df_9901['PIC'] == '')]
        cekblank    =   cekblank.loc[:,['Kode #','Nama Barang','Kategori Barang', 'PIC']].drop_duplicates()
        
        cekLainnya  =   df_9901[(df_9901['PIC'] == 'Lainnya')]
        
        cekLainnya  =   cekLainnya['Nama Barang'].drop_duplicates()
        
        df_9901 = df_9901.loc[:,['Nama Cabang','Kota/Kabupaten','Provinsi Gudang','Nomor #','Tanggal','Pemasok','Kategori Pemasok','#Group','Kode #','Nama Barang','Kategori Barang','#Purch.Qty','#Purch.UoM','#Prime.Ratio','#Prime.Qty','#Prime.UoM','#Purch.@Price','#Purch.Discount','#Prime.NetPrice','#Purch.Total','Month','Weighted_Average','Weighted_Average2','Weighted_Average Resto','Weighted_Average Resto2','PIC']]
        time_now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.download_button(
                        label="Download all Files",
                        data=df_9901.to_csv(index=False),
                        file_name=f'9901_{time_now}.csv',
                        mime='application/zip',
                    )  
