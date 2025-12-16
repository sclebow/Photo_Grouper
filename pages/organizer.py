import streamlit as st
from pathlib import Path
import shutil
from stqdm import stqdm

if "cluster_names" not in st.session_state:
    st.warning("No cluster names found in session state. Please run gallery viewer after clustering.")
    st.page_link(
        page="./pages/gallery_viewer.py",
        label="Go to Gallery Viewer Page",
    )
    st.stop()

cluster_names = st.session_state["cluster_names"]
user_directory = st.session_state["directory"]

st.subheader("Organizer Settings:")
copy_true = st.checkbox("Copy files instead of moving them", value=True)
combine_clusters_with_same_name_true = st.checkbox("Combine clusters with the same name into one folder", value=True)
run_button = st.button("Organize Photos into Folders")
st.markdown("---")

if run_button:
    images_df = st.session_state["images_df"]
    
    if combine_clusters_with_same_name_true:
        # Update cluster_names to have unique names only
        unique_label_files = {}
        unique_label_timestamps = {}  # Track min timestamp for each unique label
        items = list(cluster_names.items())
        unique_labels = set(item[1] for item in items)
        print(unique_labels)

        for key in unique_labels:
            unique_label_files[key] = []
            unique_label_timestamps[key] = None

        for cluster_label, cluster_name in items:
            cluster_data = images_df[images_df['Cluster'] == cluster_label]
            filepaths = cluster_data['Filepath'].tolist()
            unique_label_files[cluster_name].extend(filepaths)
            
            # Track the minimum timestamp for this cluster name
            min_timestamp = cluster_data['Creation Date'].min()
            if unique_label_timestamps[cluster_name] is None or min_timestamp < unique_label_timestamps[cluster_name]:
                unique_label_timestamps[cluster_name] = min_timestamp

        # Sort cluster names by their oldest timestamp (older folders first)
        sorted_cluster_names = sorted(unique_label_files.keys(), key=lambda x: unique_label_timestamps[x])

        for i, cluster_name in enumerate(sorted_cluster_names):
            filepaths = unique_label_files[cluster_name]
            # Create a subfolder in the user directory for this cluster
            cluster_folder = Path(user_directory) / f"{i+1:02d}_{cluster_name.replace(' ', '_')}"
            cluster_folder.mkdir(parents=True, exist_ok=True)

            # Move files to the cluster folder
            for filepath in stqdm(filepaths, desc=f"Processing {i+1:02d} {cluster_name}"):
                new_path = cluster_folder / Path(filepath).name
                if copy_true:
                    shutil.copy(filepath, new_path)
                    # st.write(f"Copied file: {Path(filepath).name} to {cluster_folder}")
                    print(f"Copied file: {Path(filepath).name} to {cluster_folder}")
                else:
                    shutil.move(filepath, new_path)
                    # st.write(f"Moved file: {Path(filepath).name} to {cluster_folder}")
                    print(f"Moved file: {Path(filepath).name} to {cluster_folder}")

    else:
        # Sort clusters by their oldest timestamp (older folders first)
        cluster_timestamps = {}
        for cluster_label in cluster_names.keys():
            min_timestamp = images_df[images_df['Cluster'] == cluster_label]['Creation Date'].min()
            cluster_timestamps[cluster_label] = min_timestamp
        
        sorted_clusters = sorted(cluster_names.items(), key=lambda x: cluster_timestamps[x[0]])
        
        for i, (cluster_label, cluster_name) in enumerate(sorted_clusters):
            # st.subheader(f"{cluster_name} (Cluster {cluster_label + 1})")

            filepaths = images_df[images_df['Cluster'] == cluster_label]['Filepath'].tolist()
        
            # Create a subfolder in the user directory for this cluster
            cluster_folder = Path(user_directory) / f"{i+1:02d}_{cluster_name.replace(' ', '_')}"
            cluster_folder.mkdir(parents=True, exist_ok=True)

            # Move files to the cluster folder
            for filepath in stqdm(filepaths, desc=f"Processing {i+1:02d} {cluster_name}"):
                new_path = cluster_folder / Path(filepath).name
                if copy_true:
                    shutil.copy(filepath, new_path)
                    # st.write(f"Copied file: {Path(filepath).name} to {cluster_folder}")
                    print(f"Copied file: {Path(filepath).name} to {cluster_folder}")
                else:
                    shutil.move(filepath, new_path)
                    # st.write(f"Moved file: {Path(filepath).name} to {cluster_folder}")
                    print(f"Moved file: {Path(filepath).name} to {cluster_folder}")
    
    st.success("Photo organization complete!")