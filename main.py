import streamlit as st
import os
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime
import pandas as pd
from pathlib import Path
from plotly import express as px 
from PIL import Image, ExifTags
# from stqdm import stqdm

print("\n" * 10)
print("Starting Photo Organizer Application...")

st.set_page_config(
    page_title="Fast Photo Organizer",
    page_icon="⚡",
    layout="wide",
)

st.title("Fast Photo Organizer by Creation Date")
st.subheader("Created by Scott Lebow PE")
st.write("Organize your photos quickly based on their creation dates using DBSCAN clustering.")

st.markdown("---")

user_input_page = st.Page(
    page="./pages/user_input.py",
    title="01 User Input",
)
clustering_page = st.Page(
    page="./pages/clustering.py",
    title="02 Clustering"
)
gallery_viewer_page = st.Page(
    page="./pages/gallery_viewer.py",
    title="03 Gallery Viewer"
)
organizer_page = st.Page(
    page="./pages/organizer.py",
    title="04 Organizer"
)

pg = st.navigation(
    {
        "User Input": [
            user_input_page
        ],
        "Clustering & Visualization": [
            clustering_page,
            gallery_viewer_page
        ],
        "Organizer": [
            organizer_page
        ]
    },
    position="sidebar",
)

pg.run()