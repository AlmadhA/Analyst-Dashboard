import streamlit as st
import requests
import zipfile
import io
import pandas as pd
import os
import gdown
import tempfile 


import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from urllib.request import urlopen
import json

with urlopen('https://github.com/superpikar/indonesia-geojson/blob/master/indonesia-province.json?raw=true') as response:
    ccaa = json.load(response)
# Fungsi untuk membuat map chart
def create_sales_map_chart(df):
        
    fig = px.choropleth(
        df, 
        geojson=ccaa, 
        locations='properties', 
        featureidkey="properties.Propinsi", 
        color=f'{wa_qty}',
        hover_name='properties',
        hover_data={f'{wa_qty}': True},
        color_continuous_scale='YlOrRd',
        title=f'{wa_qty}'
    )
    
    # Mengupdate peta untuk fokus ke Indonesia
    fig.update_geos(
        fitbounds="locations", 
        visible=False)
    
    # Menyesuaikan tampilan layout
    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},  # Margin minimal untuk memenuhi layar
        geo=dict(
            projection_scale=7  # Memperbesar peta
        ),
    )

    # Menampilkan map chart di Streamlit
    st.plotly_chart(fig, use_container_width=True)


# Fungsi untuk membuat barchart
def plot_grouped_barchart(df):
    fig = go.Figure()

    # Mendapatkan nama barang
    nama_barang = df['Nama Barang']
    
    # Warna berbeda untuk setiap bulan
    colors = px.colors.qualitative.Plotly

    # Menambahkan trace untuk setiap bulan
    for i, b in enumerate(bulan[-3:]):
        fig.add_trace(go.Bar(
            x=df['Nama Barang'],
            y=df[b],
            name=b,
            marker_color=colors[i % len(colors)],
            text=df[b],
            textposition='outside',
            textangle=-90,
        ))

    # Menambahkan layout
    fig.update_layout(
        title='',
        xaxis_title='NAMA BARANG',
        yaxis_title=f'{wa_qty}',
        barmode='group',  # Mengelompokkan bar per nama barang
        xaxis=dict(tickangle=-45),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Menampilkan barchart
    st.plotly_chart(fig, use_container_width=True)


# Fungsi untuk membuat line chart
def create_line_chart(df):
    # Membuat trace untuk Sales
    trace = go.Scatter(
        x=df.index,
        y=df.values,
        mode='lines+markers',  # Garis dengan titik marker
        name='Sales',
        line=dict(color='dodgerblue', width=2),
        marker=dict(size=8)
    )

    # Membuat layout untuk plot
    layout = go.Layout(
        title=dict(text='', x=0.5, font=dict(size=20, color='darkblue')),
        #xaxis=dict(title='MONTH', titlefont=dict(size=16, color='darkblue')),
        #yaxis=dict(title=f'{wa_qty}', titlefont=dict(size=16, color='darkblue')),
        xaxis=dict(title='MONTH'),
        yaxis=dict(title=f'{wa_qty}'),
        plot_bgcolor='white',
        xaxis_gridcolor='lightgray',
        yaxis_gridcolor='lightgray',
        hovermode='closest'
    )

    # Membuat figure dari trace dan layout
    fig = go.Figure(data=[trace], layout=layout)

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

def format_number(x):
    if x==0:
        return ''
    if isinstance(x, (int, float)):
        if x.is_integer():
            return "{:,.0f}".format(x)
        else:
            return "{:,.2f}".format(x)
    return x
    
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Fungsi untuk mereset state button
def reset_button_state():
    st.session_state.button_clicked = False
    
# Unduh file dari GitHub
col = st.columns(2)
with col[0]:
    st.title('Dashboard - Analisa Harga Barang')
with col[1]:
    tahun = st.selectbox("TAHUN:", ['2024','2025'], index=1, on_change=reset_button_state)


if 'df_9901' not in locals():
    df_9901 = pd.read_csv(f'Dataset/Analisa Harga Barang/{tahun}/9901.csv')

col = st.columns(3)
with col[0]:
    pic = st.selectbox("PIC:", ['All','RESTO','CP','WIP','LAINNYA'], index=1, on_change=reset_button_state)
with col[1]:
    cab = st.selectbox("NAMA CABANG:", ['All'] + sorted(df_9901['Nama Cabang'].unique().tolist()), index=0, on_change=reset_button_state)
with col[2]:
    kategori_barang = st.selectbox("KATEGORI BARANG:", ['All'] + sorted(df_9901['Kategori Barang'].unique().tolist()), index=sorted(df_9901['Kategori Barang'].unique().tolist()).index('10.FOOD [RM] - COM')+1, on_change=reset_button_state)

col = st.columns(3)
with col[0]:
    wa_qty = st.selectbox("WEIGHT AVG/QTY:", ['WEIGHT AVG (PRICE)','QUANTITY'], index= 0, on_change=reset_button_state)
with col[1]:
    list_bulan = df_9901['Month'].unique().tolist()
    bulan = st.multiselect("BULAN:", list_bulan, default = list_bulan[-3:], on_change=reset_button_state)
    bulan = sorted(bulan, key=lambda x: list_bulan.index(x))

category = st.selectbox("TOP/BOTTOM:", ['TOP','BOTTOM'], index= 0, on_change=reset_button_state)

if st.button('Show'):
    st.session_state.button_clicked = True
    
#if 'filtered_df_test' not in st.session_state:   
if st.session_state.button_clicked:
        df_prov = pd.read_csv('Dataset/Master/data_provinsi.csv')
        
        db = pd.read_csv('Dataset/Master/database barang.csv')
        db = db.drop_duplicates()
        db = pd.concat([db[db['Kode #'].astype(str).str.startswith('1')].sort_values('Kode #').drop_duplicates(subset=['Kode #']),
                        db[~db['Kode #'].astype(str).str.startswith('1')]], ignore_index=True)
        
        if kategori_barang != 'All':
            df_test = df_9901[(df_9901['Kategori Barang']==kategori_barang)]
        else:
            df_test = df_9901
            

        df_test = df_test[(df_test['PIC'].isin(df_test['PIC'].unique() if pic=='All' else [pic]))].groupby(['Month', 'Nama Cabang','Kode #']).agg({'#Prime.Qty': 'sum','#Purch.Total': 'sum'}).reset_index()      
        df_test['WEIGHT AVG (PRICE)'] = df_test['#Purch.Total'].astype(float)/df_test['#Prime.Qty'].astype(float)
        df_test = df_test.rename(columns={'#Prime.Qty':'QUANTITY'}).drop(columns='#Purch.Total')
        df_test = df_test.merge(db.drop_duplicates(), how='left', on='Kode #')
        df_test['Filter Barang'] = df_test['Kode #'].astype(str) + ' - ' + df_test['Nama Barang']
        df_prov = df_test[df_test['Month']==bulan[-1]].merge(df_prov,how='left',on='Nama Cabang')

        if kategori_barang != 'All':
            df_test = df_9901[(df_9901['Kategori Barang']==kategori_barang)]
        else:
            df_test = df_9901
            
        if cab != 'All' :
            df_vendor = df_test[(df_test['Nama Cabang']==cab)&(df_test['PIC'].isin(df_test['PIC'].unique() if pic=='All' else [pic]))].groupby(['Month','Pemasok','Kategori Pemasok','Kode #']).agg({'#Prime.Qty': 'sum','#Purch.Total': 'sum','min':'min','max':'max'}).reset_index()
            df_test = df_test[(df_test['Nama Cabang']==cab)&(df_test['PIC'].isin(df_test['PIC'].unique() if pic=='All' else [pic]))].groupby(['Month','Kode #']).agg({'#Prime.Qty': 'sum','#Purch.Total': 'sum'}).reset_index()
        else:
            df_vendor = df_test[(df_test['PIC'].isin(df_test['PIC'].unique() if pic=='All' else [pic]))].groupby(['Month','Pemasok','Kategori Pemasok','Kode #']).agg({'#Prime.Qty': 'sum','#Purch.Total': 'sum','min':'min','max':'max'}).reset_index()   
            df_test = df_test[(df_test['PIC'].isin(df_test['PIC'].unique() if pic=='All' else [pic]))].groupby(['Month','Kode #']).agg({'#Prime.Qty': 'sum','#Purch.Total': 'sum'}).reset_index()        

        df_vendor['WEIGHT AVG (PRICE)'] = df_vendor['#Purch.Total'].astype(float)/df_vendor['#Prime.Qty'].astype(float)
        df_vendor = df_vendor.rename(columns={'#Prime.Qty':'QUANTITY'}).drop(columns='#Purch.Total')
        df_vendor = df_vendor.merge(db.drop_duplicates(), how='left', on='Kode #')
        df_vendor['Filter Barang'] = df_vendor['Kode #'].astype(str) + ' - ' + df_vendor['Nama Barang']
    
        df_vendor = df_vendor.groupby(['Month','Pemasok','Kategori Pemasok','Kode #','Nama Barang','Filter Barang']).agg({'QUANTITY': 'sum','WEIGHT AVG (PRICE)': 'mean','min':'min','max':'max'}).reset_index()  
        df_vendor['Month'] = pd.Categorical(df_vendor['Month'], categories=list_bulan, ordered=True)
        df_vendor = df_vendor.sort_values('Month')
        
        df_test['WEIGHT AVG (PRICE)'] = df_test['#Purch.Total'].astype(float)/df_test['#Prime.Qty'].astype(float)
        df_test = df_test.rename(columns={'#Prime.Qty':'QUANTITY'}).drop(columns='#Purch.Total')
        df_test = df_test.merge(db.drop_duplicates(), how='left', on='Kode #')
        df_test['Filter Barang'] = df_test['Kode #'].astype(str) + ' - ' + df_test['Nama Barang']
    
        df_test = df_test.groupby(['Month', 'Kode #','Nama Barang','Filter Barang']).agg({'QUANTITY': 'sum','WEIGHT AVG (PRICE)': 'mean'}).reset_index()  
        df_test['Month'] = pd.Categorical(df_test['Month'], categories=list_bulan, ordered=True)
        df_test = df_test.sort_values('Month')
        df_test = df_test.pivot(index=['Kode #','Nama Barang','Filter Barang'],columns='Month',values=wa_qty).fillna('').reset_index()
        
        if len(bulan)>=3:
            df_test[f'Diff {bulan[-3]} - {bulan[-2]}'] = df_test.apply(lambda row: 0 if ((row[bulan[-2]] == '') or (row[bulan[-3]]=='') or (row[bulan[-3]]==0)) else ((row[bulan[-2]] - row[bulan[-3]]) / row[bulan[-3]]), axis=1)
            df_test[f'Diff {bulan[-2]} - {bulan[-1]}'] = df_test.apply(lambda row: 0 if ((row[bulan[-1]] == '') or (row[bulan[-2]]=='') or (row[bulan[-2]]==0)) else ((row[bulan[-1]] - row[bulan[-2]]) / row[bulan[-2]]), axis=1)
            df_test = df_test.sort_values(df_test.columns[-1],ascending=False) 
            #df_test.loc[:,df_test.columns[-2:]] = df_test.loc[:,df_test.columns[-2:]].applymap(lambda x: f'{x*100:.2f}%')
        if len(bulan)==2:
            df_test[f'Diff {bulan[-2]} - {bulan[-1]}'] = df_test.apply(lambda row: 0 if ((row[bulan[-1]] == '') or (row[bulan[-2]]=='') or (row[bulan[-2]]==0)) else ((row[bulan[-1]] - row[bulan[-2]]) / row[bulan[-2]]), axis=1)
            df_test = df_test.sort_values(df_test.columns[-1],ascending=False)
            #df_test.loc[:,df_test.columns[-1]] = df_test.loc[:,df_test.columns[-1:]].apply(lambda x: f'{x*100:.2f}%')
        
        if category=='TOP':
            if len(bulan)>=3:
                df_test2 = df_test[(df_test[df_test.columns[-1]]>0) & (df_test[df_test.columns[-2]]>0)]
                df_test2 = df_test2.loc[((df_test2[df_test2.columns[-1]] + df_test2[df_test2.columns[-2]]) / 2).sort_values(ascending=False).index].head(10)
            if len(bulan)==2:
                df_test2 = df_test[(df_test[df_test.columns[-1]]>0)].head(10)
            
        if category=='BOTTOM':
            if len(bulan)>=3:
                df_test2 = df_test[(df_test[df_test.columns[-1]]<0) & (df_test[df_test.columns[-2]]<0)]
                df_test2 = df_test2.loc[((df_test2[df_test2.columns[-1]] + df_test2[df_test2.columns[-2]]) / 2).sort_values(ascending=True).index].head(10)
            if len(bulan)==2:
                df_test2 = df_test[(df_test[df_test.columns[-1]]<0)].head(10)       
        
    
        df_test.loc[:,[x  for x in df_test.columns if 'Diff' in x]] = df_test.loc[:,[x  for x in df_test.columns if 'Diff' in x]].applymap(lambda x: f'{x*100:.2f}%')
        if len([x  for x in df_test.columns if 'Diff' in x])>1:
            df_test = df_test.drop(columns=[df_test.columns[-2]])
        df_month = df_test[[x for x in df_test.columns if x in list_bulan]].replace('',np.nan).replace(0,np.nan).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

        df_month = df_month.mean().apply(lambda x: f'{x:.3f}')
        if wa_qty =='WEIGHT AVG (PRICE)':     
            df_test2.loc[:,[x for x in df_test2.columns if x in list_bulan]] = df_test2.loc[:,[x for x in df_test2.columns if x in list_bulan]].applymap(lambda x: f'{x:,.2f}' if isinstance(x, float) else x)
        if wa_qty =='QUANTITY':     
            df_test2.loc[:,[x for x in df_test2.columns if x in list_bulan]] = df_test2.loc[:,[x for x in df_test2.columns if x in list_bulan]].applymap(lambda x: f'{x:,.0f}' if isinstance(x, float) else x)
        st.session_state.filtered_df_month = df_month    
        st.session_state.filtered_df_test2 = df_test2
        st.session_state.filtered_df_test = df_test
        st.session_state.filtered_df_prov = df_prov
        st.session_state.wa_qty = wa_qty

if ('filtered_df_test' in st.session_state) :
    create_line_chart(st.session_state.filtered_df_month)
    plot_grouped_barchart(st.session_state.filtered_df_test2)
    prov = pd.read_csv('Dataset/Master/prov.csv')    
    barang = st.multiselect("NAMA BARANG:", ['All']+df_test.sort_values('Kode #')['Filter Barang'].unique().tolist(), default = ['All'])

    
    if 'All' in barang:
        df_test = st.session_state.filtered_df_test.drop(columns='Filter Barang')
        df_prov = st.session_state.filtered_df_prov
    if 'All' not in barang:
        df_test = st.session_state.filtered_df_test[st.session_state.filtered_df_test['Filter Barang'].isin(barang)].drop(columns='Filter Barang')
        df_prov = st.session_state.filtered_df_prov[st.session_state.filtered_df_prov['Filter Barang'].isin(barang)]

    if wa_qty =='WEIGHT AVG (PRICE)':
        df_prov = df_prov.groupby(['Provinsi'])[['WEIGHT AVG (PRICE)']].mean().reset_index()
        df_test.loc[:,[x for x in df_test.columns if x in list_bulan]] = df_test.loc[:,[x for x in df_test.columns if x in list_bulan]].applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x)

    if wa_qty =='QUANTITY':
        df_prov = df_prov.groupby(['Provinsi'])[['QUANTITY']].sum().reset_index()
        df_test.loc[:,[x for x in df_test.columns if x in list_bulan]] = df_test.loc[:,[x for x in df_test.columns if x in list_bulan]].applymap(lambda x: f'{x:.0f}' if isinstance(x, float) else x)

    df_prov['Provinsi'] = df_prov['Provinsi'].replace('BANTEN','PROBANTEN')
    #create_sales_map_chart(prov.merge(df_prov,how='left',left_on='properties',right_on='Provinsi').drop(columns='Provinsi').fillna(0))
    df_test.iloc[:,2:-1] = df_test.iloc[:,2:-1].replace('',0).astype(float)
    df_test = df_test.style.format(lambda x: format_number(x)).background_gradient(cmap='Reds', axis=1, subset=df_test.columns[2:-1])
    st.dataframe(df_test, use_container_width=True, hide_index=True)
    
    vendor = st.multiselect("PEMASOK:", ['All']+df_vendor[df_vendor['Filter Barang'].isin(df_vendor['Filter Barang'].unique() if barang==['All'] else barang)].sort_values('Pemasok')['Pemasok'].unique().tolist(), default = ['All'])
    
    df_test = df_vendor[(df_vendor['Filter Barang'].isin(df_vendor['Filter Barang'].unique() if barang==['All'] else barang)) & (df_vendor['Pemasok'].isin(df_vendor['Pemasok'].unique() if vendor==['All'] else vendor))]
    df_test = df_test.pivot(index=['Pemasok','Kode #','Nama Barang'],columns='Month',values=wa_qty).fillna('').reset_index()

    if len(bulan)>=3:
        df_test[f'Diff {bulan[-3]} - {bulan[-2]}'] = df_test.apply(lambda row: 0 if ((row[bulan[-2]] == '') or (row[bulan[-3]]=='') or (row[bulan[-3]]==0)) else ((row[bulan[-2]] - row[bulan[-3]]) / row[bulan[-3]]), axis=1)
        df_test[f'Diff {bulan[-2]} - {bulan[-1]}'] = df_test.apply(lambda row: 0 if ((row[bulan[-1]] == '') or (row[bulan[-2]]=='') or (row[bulan[-2]]==0)) else ((row[bulan[-1]] - row[bulan[-2]]) / row[bulan[-2]]), axis=1)
        df_test = df_test.sort_values(df_test.columns[-1],ascending=False) 
        #df_test.loc[:,df_test.columns[-2:]] = df_test.loc[:,df_test.columns[-2:]].applymap(lambda x: f'{x*100:.2f}%')
    if len(bulan)==2:
        df_test[f'Diff {bulan[-2]} - {bulan[-1]}'] = df_test.apply(lambda row: 0 if ((row[bulan[-1]] == '') or (row[bulan[-2]]=='') or (row[bulan[-2]]==0)) else ((row[bulan[-1]] - row[bulan[-2]]) / row[bulan[-2]]), axis=1)
        df_test = df_test.sort_values(df_test.columns[-1],ascending=False)
        #df_test.loc[:,df_test.columns[-1]] = df_test.loc[:,df_test.columns[-1:]].apply(lambda x: f'{x*100:.2f}%')   

    if wa_qty =='WEIGHT AVG (PRICE)':
        df_test.loc[:,[x for x in df_test.columns if x in list_bulan]] = df_test.loc[:,[x for x in df_test.columns if x in list_bulan]].applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x)

    if wa_qty =='QUANTITY':
        df_test.loc[:,[x for x in df_test.columns if x in list_bulan]] = df_test.loc[:,[x for x in df_test.columns if x in list_bulan]].applymap(lambda x: f'{x:.0f}' if isinstance(x, float) else x)
    
    df_test.loc[:,[x  for x in df_test.columns if 'Diff' in x]] = df_test.loc[:,[x  for x in df_test.columns if 'Diff' in x]].applymap(lambda x: f'{x*100:.2f}%')
    if len([x  for x in df_test.columns if 'Diff' in x])>1:
        df_test = df_test.drop(columns=[df_test.columns[-2]])
    #st.dataframe(df_test)
    df_test.iloc[:,3:-1] = df_test.iloc[:,3:-1].replace('',0).astype(float)
    df_test = df_test.style.format(lambda x:format_number(x)).background_gradient(cmap='Reds', axis=1, subset=df_test.columns[3:-1])
    st.dataframe(df_test, use_container_width=True, hide_index=True)

#    if barang !=['All']:
    kat_vendor = st.multiselect("KATEGORI PEMASOK:", ['All']+df_vendor[df_vendor['Filter Barang'].isin(df_vendor['Filter Barang'].unique() if barang==['All'] else barang)].sort_values('Kategori Pemasok')['Kategori Pemasok'].unique().tolist(), default = ['All'])
    
    df_vendor = df_vendor[(df_vendor['Filter Barang'].isin(df_vendor['Filter Barang'].unique() if barang==['All'] else barang)) & (df_vendor['Kategori Pemasok'].isin(df_vendor['Kategori Pemasok'].unique() if kat_vendor==['All'] else kat_vendor))]
    df_vendor = df_vendor.groupby(['Month','Kategori Pemasok','Kode #','Nama Barang']).agg({'min':'min','max':'max'}).reset_index().dropna(subset=['min','max'])  
    df_vendor['Month'] = pd.Categorical(df_vendor['Month'], categories=list_bulan, ordered=True)
    df_vendor = df_vendor.sort_values('Month')
    df_vendor[f'Diff Min-Max {bulan[-1]}'] =  ((df_vendor['max']-df_vendor['min'])/df_vendor['min'])
    df_vendor['Range'] = df_vendor['min'].apply(lambda x: f'{x:,.2f}').fillna('') + '-' + df_vendor['max'].apply(lambda x: f'{x:,.2f}').fillna('')
    
    df_vendor = df_vendor.pivot(index=['Kategori Pemasok','Kode #','Nama Barang'], columns='Month',values='Range').fillna('').reset_index().sort_values(['Kode #','Kategori Pemasok']).merge(
                df_vendor[df_vendor['Month']==bulan[-1]][['Kategori Pemasok','Kode #',f'Diff Min-Max {bulan[-1]}']].dropna(subset=f'Diff Min-Max {bulan[-1]}'), how='left').sort_values(f'Diff Min-Max {bulan[-1]}', ascending=False)
    df_vendor[f'Diff Min-Max {bulan[-1]}'] = df_vendor[f'Diff Min-Max {bulan[-1]}'].fillna(0).apply(lambda x: f'{x*100:.2f}%')
    st.dataframe(df_vendor, use_container_width=True, hide_index=True)

