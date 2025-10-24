import streamlit as st
import os
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime
import pandas as pd
from pathlib import Path
from plotly import express as px 
from PIL import Image, ExifTags
from stqdm import stqdm

print("\n" * 10)
print("Starting Photo Organizer Application...")

st.set_page_config(
    page_title="Fast Photo Organizer",
    page_icon="⚡",
    layout="wide",
)

# Prompt the user for a directory containing photos
st.title("⚡ Fast Photo Organizer by Creation Date")

# Add configuration options in sidebar
st.sidebar.header("Configuration")

directory = st.text_input("Enter the directory path containing your photos:", 
                         help="Full path to the folder containing your image files")

if not directory:
    st.info("Please enter a directory path to proceed.")
    st.stop()

# Verify the directory exists
if not os.path.isdir(directory):
    st.error("The specified directory does not exist. Please enter a valid directory path.")
    st.stop()

# Get all images from the specified directory (optimized for speed)
@st.cache_data
def get_images_metadata(directory):
    """Extract only metadata from image files without loading full image data"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    images_data = []
    
    # Use pathlib for more efficient directory traversal
    directory_path = Path(directory)

    for filepath in directory_path.iterdir():
        if filepath.is_file() and filepath.suffix.lower() in image_extensions:
            try:
                # Get file stats in one call
                stat = filepath.stat()
                file_size = stat.st_size
                image = Image.open(filepath)
                exif = {ExifTags.TAGS[k]: v for k, v in image._getexif().items()} if image._getexif() else {}
                modification_time = exif.get('DateTimeOriginal')
                try:
                    modification_time = datetime.strptime(modification_time, '%Y:%m:%d %H:%M:%S')
                except (ValueError, TypeError):
                    modification_time = None
                    print("Could not parse modification time for file:", filepath)
                
                images_data.append({
                    "Filename": filepath.name,
                    "Filepath": str(filepath),
                    "Filetype": f"image/{filepath.suffix[1:].lower()}",
                    "Filesize": file_size,
                    "Creation Date": modification_time
                })
            except (OSError, PermissionError):
                # Skip files that can't be accessed
                continue
    
    return images_data

# Add progress indicator for large directories
with st.spinner('Scanning directory for image files...'):
    images_data = get_images_metadata(directory)

# Check if any images were found
if not images_data:
    st.warning("No image files found in the specified directory.")
    st.stop()

# Display file count for user feedback
st.info(f"Found {len(images_data)} image files. Processing...")

# Process the image metadata efficiently
df = pd.DataFrame(images_data)
nan_df = df[df["Creation Date"].isna()]
df = df.dropna(subset=["Creation Date"])

st.info(f"Processing {len(df)} images with valid creation dates. {len(nan_df)} images missing creation dates. View details in the 'See Raw Data' section below.")

fig = px.histogram(
    df,
    x = 'Creation Date',
    nbins=20,
)

st.plotly_chart(fig)

with st.expander("See Raw Data"):
    st.write("Photos with missing creation dates:")
    st.dataframe(nan_df, use_container_width=True)
    st.write("Photos with valid creation dates:")
    st.dataframe(df, use_container_width=True)

# Convert creation dates to timestamps for clustering
timestamps = np.array([data["Creation Date"] for data in df.to_dict('records')])
timestamps = timestamps.reshape(-1, 1)
timestamps = np.array([[dt.timestamp()] for dt in timestamps.flatten()])

# Apply DBSCAN clustering (convert hours to seconds)
grouping_hours = st.sidebar.slider("Group photos taken within how many hours?", 
                                    min_value=0.01, max_value=1.0, value=0.01, step=0.01,
                                    help="Photos taken within this number of hours will be grouped together.")
eps_seconds = grouping_hours * 3600

clustering = DBSCAN(eps=eps_seconds, min_samples=1).fit(timestamps)
labels = clustering.labels_

# Add cluster information to dataframe
df["Cluster"] = labels

# Sort by creation date for better organization
df = df.sort_values('Creation Date')

# Display summary statistics
st.subheader("Clustering Summary")
cluster_counts = df.groupby('Cluster').size().reset_index(name='Photo Count')
st.write(f"Total photos: {len(df)}")
st.write(f"Number of groups: {len(cluster_counts)}")

# Display grouped files as a dataframe
st.subheader("Photo Groups")

# Add download option for CSV
csv = df.to_csv(index=False)
st.download_button(
    label="Download results as CSV",
    data=csv,
    file_name=f"photo_groups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

st.dataframe(df, use_container_width=True)

# Show cluster data
st.subheader("Cluster Data")

# Show cluster distribution
st.subheader("Time Group Distribution")

# Create a histogram with the x-axis as timestamps and the y-axis as counts
resolution = st.slider("Select histogram resolution (number of bins multiplier):", 
                           min_value=1, max_value=20, value=5, step=1)

# Create stacked bars by cluster
fig = px.histogram(
        df, 
        x='Creation Date', 
        color='Cluster', 
        barmode='stack', 
        title="Photo Creation Time Distribution by Cluster",
        nbins=len(cluster_counts) * resolution
    )

st.plotly_chart(fig)

# Create a dictionary of filepaths for each cluster
filepaths_by_cluster = {}
for cluster_label in df['Cluster'].unique():
    filepaths = df[df['Cluster'] == cluster_label]['Filepath'].tolist()
    filepaths_by_cluster[cluster_label] = filepaths

my_bar = st.progress(0, text='Loading and displaying images...')

tabs = st.tabs([f"Cluster {i + 1}" for i in range(len(filepaths_by_cluster))])
num_columns_per_tab = 5

column_dict = {}
for i, tab in enumerate(tabs):
    with tab:
        cols = st.columns(num_columns_per_tab)
        column_dict[i] = cols

longest_filepath_list_length = max(len(filepaths) for filepaths in filepaths_by_cluster.values())

print("Longest filepath list length:", longest_filepath_list_length)

number_of_images_to_load_together = 10

images_loaded = 0
# Load and display images in batches to optimize performance
for batch_start in range(0, longest_filepath_list_length, number_of_images_to_load_together):
    for cluster_label, filepaths in filepaths_by_cluster.items():
        tab_index = cluster_label
        cols = column_dict[tab_index]
        
        batch_filepaths = filepaths[batch_start:batch_start + number_of_images_to_load_together]
        
        for idx, filepath in enumerate(batch_filepaths):
            col = cols[idx % num_columns_per_tab]
            try:
                image = Image.open(filepath)
                col.image(image, caption=Path(filepath).name)
            except Exception as e:
                col.write(f"Error loading image: {Path(filepath).name}")
            images_loaded += 1
            progress_percentage = int((images_loaded / len(df)) * 100)
            my_bar.progress(progress_percentage, text=f'Batch Loaded {images_loaded} of {len(df)} images. Currently loading Cluster {cluster_label + 1}.')

my_bar.progress(100, text='All images loaded and displayed.')
my_bar.empty()