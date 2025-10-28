import streamlit as st
from pathlib import Path
from PIL import Image
from stqdm import stqdm

# Performance optimization settings
st.header("Cluster Image Viewer Settings")
thumbnail_size = st.selectbox("Thumbnail size:", [150, 200, 300, 400], index=1, 
                                     help="Smaller thumbnails load faster")
num_columns = st.slider("Number of columns to display images:", 
                                min_value=2, max_value=30, value=15, step=1,
                                help="Adjust number of columns for image display")

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
    cols = st.columns(num_columns)
    
    # Display all images in columns with progress bar
    for idx, filepath in stqdm(enumerate(filepaths), 
                              desc=f"Loading images for {st.session_state['cluster_names'].get(cluster_label, f'Group {cluster_label + 1}')}",
                              total=len(filepaths)):
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
unique_clusters = st.session_state["images_df"]['Cluster'].unique()
for cluster_label in stqdm(unique_clusters, desc="Organizing clusters"):
    filepaths = st.session_state["images_df"][st.session_state["images_df"]['Cluster'] == cluster_label]['Filepath'].tolist()
    filepaths_by_cluster[cluster_label] = filepaths

# Display cluster information
st.subheader("Photo Clusters")
total_images = sum(len(filepaths) for filepaths in filepaths_by_cluster.values())
st.info(f"Total images: {total_images} | Clusters: {len(filepaths_by_cluster)}")

# Create tabs for each cluster
if filepaths_by_cluster:
    cluster_labels = sorted(filepaths_by_cluster.keys())
    
    if "cluster_names" not in st.session_state:
        st.session_state["cluster_names"] = {label: f"Group {i + 1}" for i, label in enumerate(cluster_labels)}

    tab_labels = [st.session_state["cluster_names"][label] for label in cluster_labels]
    
    tabs = st.tabs(tab_labels)
    
    for i, (tab, cluster_label) in enumerate(zip(tabs, cluster_labels)):

        with tab:
            new_name = st.text_input(
                label=f"Rename Cluster: ",
                value=st.session_state["cluster_names"][cluster_label],
                key=f"rename_cluster_{i}"
            )

            if new_name != st.session_state["cluster_names"][cluster_label]:
                st.session_state["cluster_names"][cluster_label] = new_name
                print(f"Renamed cluster {cluster_label} to {new_name}")
                st.rerun()

            filepaths = filepaths_by_cluster[cluster_label]
            
            # Display all images for this cluster
            display_images_in_tab(cluster_label, filepaths)
else:
    st.warning("No image clusters found.")