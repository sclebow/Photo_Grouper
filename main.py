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

# Prompt the user for a directory containing photos
st.title("⚡ Fast Photo Organizer by Creation Date")

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
    st.dataframe(nan_df, width='stretch')
    st.write("Photos with valid creation dates:")
    st.dataframe(df, width='stretch')

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

st.dataframe(df, width='stretch')

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

# Performance optimization settings
st.sidebar.header("Performance Settings")
thumbnail_size = st.sidebar.selectbox("Thumbnail size:", [150, 200, 300, 400], index=1, 
                                     help="Smaller thumbnails load faster")

@st.cache_data
def load_and_resize_image(filepath, size=(200, 200)):
    """Load and resize image to thumbnail size for faster display"""
    try:
        with Image.open(filepath) as image:
            # Convert to RGB if necessary (for consistency)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Create thumbnail while maintaining aspect ratio
            image.thumbnail(size, Image.Resampling.LANCZOS)
            return image.copy()
    except Exception as e:
        return None

def display_images_in_tab(cluster_label, filepaths):
    """Display all images for a specific cluster"""
    if not filepaths:
        st.write("No images in this cluster.")
        return
    
    # Create columns for layout
    num_columns = 5
    cols = st.columns(num_columns)
    
    # Display all images in columns
    for idx, filepath in enumerate(filepaths):
        col = cols[idx % num_columns]
        with col:
            # Load and display thumbnail
            thumbnail = load_and_resize_image(filepath, (thumbnail_size, thumbnail_size))
            if thumbnail:
                st.image(thumbnail, caption=Path(filepath).name, width='stretch')
                # Add button to view full size
                if st.button(f"View Full Size", key=f"full_{cluster_label}_{idx}"):
                    with st.expander("Full Size Image", expanded=True):
                        full_image = Image.open(filepath)
                        st.image(full_image, caption=Path(filepath).name)
            else:
                st.error(f"Error loading: {Path(filepath).name}")

# Create a dictionary of filepaths for each cluster
filepaths_by_cluster = {}
for cluster_label in df['Cluster'].unique():
    filepaths = df[df['Cluster'] == cluster_label]['Filepath'].tolist()
    filepaths_by_cluster[cluster_label] = filepaths

# Display cluster information
st.subheader("Photo Clusters")
total_images = sum(len(filepaths) for filepaths in filepaths_by_cluster.values())
st.info(f"Total images: {total_images} | Clusters: {len(filepaths_by_cluster)}")

# Create tabs for each cluster
if filepaths_by_cluster:
    cluster_labels = sorted(filepaths_by_cluster.keys())
    tab_labels = [f"Cluster {i + 1} ({len(filepaths_by_cluster[label])} photos)" 
                  for i, label in enumerate(cluster_labels)]
    
    tabs = st.tabs(tab_labels)
    
    for i, (tab, cluster_label) in enumerate(zip(tabs, cluster_labels)):
        with tab:
            filepaths = filepaths_by_cluster[cluster_label]
            
            # Display all images for this cluster
            with st.spinner(f'Loading images for Cluster {cluster_label + 1}...'):
                display_images_in_tab(cluster_label, filepaths)
else:
    st.warning("No image clusters found.")