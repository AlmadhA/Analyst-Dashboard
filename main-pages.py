import streamlit as st
import pandas as pd
import requests

page_1 = st.Page("Pages/Dashboard/Leadtime Internal and Eksternal.py", title="Leadtime Internal & Eksternal")
page_3 = st.Page("Pages/Dashboard/Analisa Harga Barang.py", title="Analisa Harga Barang")
page_4 = st.Page("Pages/Dashboard/Selisih Ojol.py", title="Selisih Ojol")
page_9 = st.Page("Pages/Dashboard/Test.py", title="(Maintenance)")

page_5 = st.Page("Pages/Tools/GIS-Cleaning & Rekap SCM.py", title="GIS Cleaning & Rekap SCM")
page_6 = st.Page("Pages/Tools/ABO-DA.py", title="Automate Breakdown Ojol")

pg = st.navigation({'Dashboard':[
                    page_1, page_3, page_4],
                   'Tools':[
                    page_5,page_6
                   ]},expanded=True)
pg.run()



