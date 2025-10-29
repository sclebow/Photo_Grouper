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
    if combine_clusters_with_same_name_true:
        # Update cluster_names to have unique names only
        unique_label_files = {}
        items = list(cluster_names.items())
        unique_labels = set(item[1] for item in items)
        print(unique_labels)

        for key in unique_labels:
            unique_label_files[key] = []

        for cluster_label, cluster_name in items:
            filepaths = st.session_state["images_df"][st.session_state["images_df"]['Cluster'] == cluster_label]['Filepath'].tolist()
            unique_label_files[cluster_name].extend(filepaths)

        for i, (cluster_name, filepaths) in enumerate(unique_label_files.items()):
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
        for i, (cluster_label, cluster_name) in enumerate(cluster_names.items()):
            # st.subheader(f"{cluster_name} (Cluster {cluster_label + 1})")

            filepaths = st.session_state["images_df"][st.session_state["images_df"]['Cluster'] == cluster_label]['Filepath'].tolist()
        
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