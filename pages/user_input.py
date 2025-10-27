import streamlit as st
import os
from pathlib import Path
from PIL import Image, ExifTags
import pandas as pd
from datetime import datetime
from plotly import express as px
from stqdm import stqdm

# Add configuration options in sidebar
st.sidebar.header("Configuration")

if "directory" not in st.session_state:
    st.session_state["directory"] = None

directory = st.text_input(
    "Enter the directory path containing your photos:", 
    help="Full path to the folder containing your image files",
    value=st.session_state["directory"]
)

if not directory:
    st.info("Please enter a directory path to proceed.")
    st.stop()

# Verify the directory exists
if not os.path.isdir(directory):
    st.error("The specified directory does not exist. Please enter a valid directory path.")
    st.stop()

if directory == st.session_state.get("directory", None):
    st.info("Directory unchanged. Using cached data.")

else:
    st.session_state["directory"] = directory

    # Get all images from the specified directory (optimized for speed)
    def get_images_metadata(directory):
        """Extract only metadata from image files without loading full image data"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        images_data = []
        
        # Use pathlib for more efficient directory traversal
        directory_path = Path(directory)

        for filepath in stqdm(directory_path.iterdir(), desc="Scanning files"):
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
    st.session_state["images_data"] = get_images_metadata(directory)

    # Check if any images were found
    if not st.session_state["images_data"]:
        st.warning("No image files found in the specified directory.")
        st.stop()

    # Display file count for user feedback
    st.info(f"Found {len(st.session_state['images_data'])} image files. Processing...")

    # Process the image metadata efficiently
    df = pd.DataFrame(st.session_state["images_data"])
    nan_df = df[df["Creation Date"].isna()]
    df = df.dropna(subset=["Creation Date"])

    st.info(f"Processing {len(df)} images with valid creation dates. {len(nan_df)} images missing creation dates. View details in the 'See Raw Data' section below.")

    st.session_state["images_df"] = df
    st.session_state["nan_images_df"] = nan_df

fig = px.histogram(
    st.session_state["images_df"],
    x = 'Creation Date',
    nbins=20,
)

st.plotly_chart(fig)

with st.expander("See Raw Data"):
    st.write("Photos with missing creation dates:")
    st.dataframe(st.session_state["nan_images_df"], width='stretch')
    st.write("Photos with valid creation dates:")
    st.dataframe(st.session_state["images_df"], width='stretch')

# Add a link to the clustering page
st.markdown("---")
st.page_link(
    page="./pages/clustering.py",
    label="Proceed to Clustering Page",
)