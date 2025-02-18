import streamlit as st
import pandas as pd
import requests

page_1 = st.Page("Pages/Dashboard/Leadtime Internal and Eksternal.py", title="Leadtime Internal & Eksternal")
page_2 = st.Page("Pages/Dashboard/Inventory Stock Monitoring.py", title="Inventory Stock Monitoring")
page_3 = st.Page("Pages/Dashboard/Safety Stock.py", title="Safety Stock")
page_4 = st.Page("Pages/Dashboard/Analisa Harga Barang.py", title="Analisa Harga Barang")
page_5 = st.Page("Pages/Dashboard/Selisih Ojol.py", title="Selisih Ojol")
page_9 = st.Page("Pages/Dashboard/Test.py", title="(Maintenance)")

page_6 = st.Page("Pages/Tools/GIS-Cleaning & Rekap SCM.py", title="GIS Cleaning & Rekap SCM")
page_7 = st.Page("Pages/Tools/ABO-DA.py", title="Automate Breakdown Ojol")
page_8 = st.Page("Pages/Dashboard/Weight Average (9901).py", title="Weight Average 99.01")

pg = st.navigation({'Dashboard':[
                    page_1, page_2, page_3, page_4, page_5],
                   'Tools':[
                    page_6,page_7,page_8
                   ]},expanded=True)
pg.run()



