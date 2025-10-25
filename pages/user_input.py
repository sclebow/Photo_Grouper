import streamlit as st
import os

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

st.session_state["directory"] = directory