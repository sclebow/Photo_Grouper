import streamlit as st
import os
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime
import pandas as pd
from pathlib import Path

# Prompt the user for a directory containing photos
st.title("⚡ Fast Photo Organizer by Creation Date")

# Add configuration options in sidebar
st.sidebar.header("Configuration")
grouping_hours = st.sidebar.slider("Group photos within (hours):", 1, 24, 1, 
                                   help="Photos taken within this time window will be grouped together")

directory = st.text_input("Enter the directory path containing your photos:", 
                         help="Full path to the folder containing your image files")

if directory:
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
                    modification_time = stat.st_mtime
                    
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

    # Convert creation dates to timestamps for clustering
    timestamps = np.array([data["Creation Date"] for data in images_data])
    timestamps = timestamps.reshape(-1, 1)

    # Apply DBSCAN clustering (convert hours to seconds)
    eps_seconds = grouping_hours * 3600
    clustering = DBSCAN(eps=eps_seconds, min_samples=1).fit(timestamps)
    labels = clustering.labels_

    # Add cluster information to dataframe
    df["Cluster"] = labels

    # Convert timestamps to readable dates for display
    df["Creation Date"] = pd.to_datetime(df["Creation Date"], unit='s')

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

    # Show cluster distribution
    st.subheader("Group Distribution")
    st.bar_chart(cluster_counts.set_index('Cluster'))