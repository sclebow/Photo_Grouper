import streamlit as st
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime
import pandas as pd
from pathlib import Path
from plotly import express as px
from PIL import Image, ExifTags

if "images_df" not in st.session_state:
    st.warning("No image data found. Please upload images on the User Input page.")
    st.page_link(
        page="./pages/user_input.py",
        label="Go to User Input Page",
    )
    st.stop()

# Convert creation dates to timestamps for clustering
timestamps = np.array([data["Creation Date"] for data in st.session_state["images_df"].to_dict('records')])
timestamps = timestamps.reshape(-1, 1)
timestamps = np.array([[dt.timestamp()] for dt in timestamps.flatten()])

# Apply DBSCAN clustering (convert hours to seconds)
grouping_minutes = st.slider("Group photos taken within how many minutes?", 
                                    min_value=0.05, max_value=10.0, value=st.session_state.get("grouping_minutes", 1.0), step=0.05,
                                    help="Photos taken within this number of minutes will be grouped together.")

st.session_state["grouping_minutes"] = grouping_minutes

eps_seconds = grouping_minutes * 60

clustering = DBSCAN(eps=eps_seconds, min_samples=1).fit(timestamps)
labels = clustering.labels_

# Add cluster information to dataframe
st.session_state["images_df"]["Cluster"] = labels

# Sort by creation date for better organization
st.session_state["images_df"] = st.session_state["images_df"].sort_values('Creation Date')

# Display summary statistics
st.subheader("Clustering Summary")
cluster_counts = st.session_state["images_df"].groupby('Cluster').size().reset_index(name='Photo Count')
st.write(f"Total photos: {len(st.session_state['images_df'])}")
st.write(f"Number of groups: {len(cluster_counts)}")

# Display grouped files as a dataframe
st.subheader("Photo Groups")

# Reorder clusters by creation date
cluster_order = st.session_state["images_df"].groupby('Cluster')['Creation Date'].min().sort_values().index.tolist()
cluster_name_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_order)}

st.session_state["images_df"]['Cluster'] = st.session_state["images_df"]['Cluster'].map(cluster_name_mapping)

with st.expander("Cluster Data"):
    # Add download option for CSV
    csv = st.session_state["images_df"].to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name=f"photo_groups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    st.dataframe(st.session_state["images_df"], width='stretch')

# Show cluster data
st.subheader("Cluster Data")

# Show cluster distribution
st.subheader("Time Group Distribution")

# Create a histogram with the x-axis as timestamps and the y-axis as counts
resolution = st.slider("Select histogram resolution (number of bins multiplier):", 
                           min_value=1, max_value=20, value=st.session_state.get("histogram_resolution", 5), step=1)

st.session_state["histogram_resolution"] = resolution

# Create stacked bars by cluster
fig = px.histogram(
        st.session_state["images_df"], 
        x='Creation Date', 
        color='Cluster', 
        barmode='stack', 
        title="Photo Creation Time Distribution by Cluster",
        nbins=len(cluster_counts) * resolution
    )

st.plotly_chart(fig)

# Remove "cluster_names" from session state if it exists
if "cluster_names" in st.session_state:
    # Check if the cluster names length matches current clusters
    current_clusters = st.session_state["images_df"]['Cluster'].unique()
    if len(current_clusters) != len(st.session_state["cluster_names"]):
        del st.session_state["cluster_names"]