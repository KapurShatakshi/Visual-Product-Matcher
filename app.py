# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import joblib
# import os
# import requests
# from PIL import Image
# import io
# import math
# from sklearn.neighbors import NearestNeighbors
# import base64
# import re

# st.set_page_config(
#     page_title="Visual Product Matcher",
#     page_icon="👕",
#     layout="wide"
# )

# ARTIFACTS_DIR = "." 
# CSV_FILE = "styles_deployment_sample.csv"
# EMBEDDINGS_FILE = "emb_norm_sample.npy"
# IMAGE_DIR = "images"

# @st.cache_resource
# def load_artifacts():
#     try:
#         extractor = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, "feature_extractor.keras"))
        
#         # Load the sample CSV and construct the full image paths
#         df = pd.read_csv(os.path.join(ARTIFACTS_DIR, CSV_FILE))
#         df['image'] = df['id'].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg"))

#         kmeans = joblib.load(os.path.join(ARTIFACTS_DIR, "kmeans.joblib"))
#         emb_norm = np.load(os.path.join(ARTIFACTS_DIR, EMBEDDINGS_FILE))
        
#         # Calculate centroids from the full kmeans model
#         centroids = kmeans.cluster_centers_
#         centroids_norm = centroids.astype("float32") / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        
#         # Dynamically create cluster mappings from the loaded sample data
#         unique_clusters = sorted(df['cluster'].unique())
#         cluster_to_idx_map = {c: df[df['cluster'] == c].index.to_numpy() for c in unique_clusters}

#         cluster_to_nn = {}
#         for cluster_id, indices in cluster_to_idx_map.items():
#             if len(indices) > 1:
#                 cluster_embeddings = emb_norm[indices]
#                 nn_model = NearestNeighbors(n_neighbors=min(20, len(indices)), metric="cosine", algorithm='brute')
#                 nn_model.fit(cluster_embeddings)
#                 cluster_to_nn[cluster_id] = nn_model
#             else:
#                 cluster_to_nn[cluster_id] = None # Handle small clusters

#         return extractor, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn

#     except FileNotFoundError as e:
#         st.error(f"Error loading artifact: {e}. Please ensure all required files are in the GitHub repository.")
#         st.stop()

# def load_and_preprocess_image(image_source, source_type='file'):
#     try:
#         if source_type == 'url':
#             # --- Case 1: Base64 data URI ---
#             if image_source.startswith("data:image"):
#                 match = re.match(r"data:image\/[a-zA-Z]+;base64,(.*)", image_source)
#                 if match:
#                     image_data = base64.b64decode(match.group(1))
#                     image = Image.open(io.BytesIO(image_data)).convert("RGB")
#                 else:
#                     raise ValueError("Invalid base64 image string.")
            
#             # --- Case 2: Google Drive link ---
#             elif "drive.google.com" in image_source:
#                 # Convert to direct download
#                 file_id_match = re.search(r"/d/([a-zA-Z0-9_-]+)", image_source)
#                 if file_id_match:
#                     file_id = file_id_match.group(1)
#                     image_source = f"https://drive.google.com/uc?id={file_id}"
#                 else:
#                     raise ValueError("Invalid Google Drive link format.")
#                 response = requests.get(image_source, stream=True, timeout=10)
#                 response.raise_for_status()
#                 image = Image.open(io.BytesIO(response.content)).convert("RGB")

#             elif "github.com" in image_source and not "raw.githubusercontent.com" in image_source:
#                 image_source = image_source.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
#                 response = requests.get(image_source, stream=True, timeout=10)
#                 response.raise_for_status()
#                 image = Image.open(io.BytesIO(response.content)).convert("RGB")

#             else:
#                 headers = {'User-Agent': 'Mozilla/5.0'}
#                 response = requests.get(image_source, headers=headers, stream=True, timeout=10)
#                 response.raise_for_status()
#                 image = Image.open(io.BytesIO(response.content)).convert("RGB")

#         else: 
#             image = Image.open(image_source).convert("RGB")

#         image_display = image.copy()
#         image = image.resize((224, 224))
#         img_array = tf.keras.preprocessing.image.img_to_array(image)
#         img_array_expanded = np.expand_dims(img_array, axis=0)
#         return tf.keras.applications.resnet50.preprocess_input(img_array_expanded), image_display

#     except Exception as e:
#         st.error(f"Could not load image. Error: {e}")
#         return None, None

# def extract_embedding(image_tensor, extractor_model):
#     embedding = extractor_model.predict(image_tensor, verbose=0)[0]
#     embedding = embedding.astype("float32")
#     embedding /= (np.linalg.norm(embedding) + 1e-10)
#     return embedding

# def search_similar(query_embedding, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn, top_k=12):
#     dists_to_centroids = 1.0 - (centroids_norm @ query_embedding)
#     closest_cluster = int(np.argmin(dists_to_centroids))
    
#     nn_cluster = cluster_to_nn.get(closest_cluster)
#     idx_pool = cluster_to_idx_map.get(closest_cluster, np.array([]))

#     if nn_cluster is None or len(idx_pool) == 0:
#         return pd.DataFrame() 

#     dist, ind = nn_cluster.kneighbors(query_embedding.reshape(1, -1), n_neighbors=min(top_k, len(idx_pool)))
    
#     hits_indices = idx_pool[ind[0]]
#     distances = dist[0]
    
#     results_df = df.iloc[hits_indices].copy()
#     results_df['cosine_distance'] = distances
#     results_df['similarity_score'] = 1 - results_df['cosine_distance']
#     return results_df.sort_values('cosine_distance')

# st.title("👕 Visual Product Matcher")
# st.write("Upload an image or provide a URL to find visually similar fashion products.")

# if 'search_results' not in st.session_state:
#     st.session_state.search_results = None
# if 'uploaded_image_display' not in st.session_state:
#     st.session_state.uploaded_image_display = None

# with st.spinner("Loading models and catalog..."):
#     extractor, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn = load_artifacts()

# st.sidebar.header("Search Options")
# input_method = st.sidebar.radio("Input method:", ("Upload an Image", "Enter Image URL"))

# query_image, uploaded_image_display = None, None

# newly_uploaded = None

# if input_method == "Upload an Image":
#     uploaded_file = st.sidebar.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"])
#     if uploaded_file:
#         newly_uploaded = uploaded_file.name
#         query_image, uploaded_image_display = load_and_preprocess_image(uploaded_file, source_type='file')
# else:
#     url = st.sidebar.text_input("Enter image URL:")
#     if url:
#         newly_uploaded = url
#         query_image, uploaded_image_display = load_and_preprocess_image(url, source_type='url')

# if newly_uploaded != st.session_state.get('last_uploaded_name'):
#     st.session_state.search_results = None
#     st.session_state.last_uploaded_name = newly_uploaded

# col1, col2 = st.columns([1, 3])

# with col1:
#     if uploaded_image_display:
#         st.session_state.uploaded_image_display = uploaded_image_display
#         st.header("Your Image")
#         st.image(uploaded_image_display, use_container_width=True)
        
#         if st.button("Find Similar Products", type="primary"):
#             with st.spinner("Searching for matches..."):
#                 query_embedding = extract_embedding(query_image, extractor)
#                 st.session_state.search_results = search_similar(query_embedding, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn)
#     else:
#         st.session_state.uploaded_image_display = None
#         st.session_state.search_results = None

# with col2:
#     st.header("Similar Products")
#     if st.session_state.search_results is not None:
#         results = st.session_state.search_results
        
#         min_similarity = st.slider("Filter by Similarity Score:", 0.0, 1.0, 0.0, 0.01)
        
#         filtered_results = results[results['similarity_score'] >= min_similarity]

#         if filtered_results.empty:
#             st.warning("No results match the current filter. Try lowering the similarity score.")
#         else:
#             num_cols = 4
#             num_rows = math.ceil(len(filtered_results) / num_cols)
#             for i in range(num_rows):
#                 row_cols = st.columns(num_cols)
#                 for j in range(num_cols):
#                     idx = i * num_cols + j
#                     if idx < len(filtered_results):
#                         product = filtered_results.iloc[idx]
#                         with row_cols[j]:
#                             image_path = product.get('image', '')
#                             if os.path.exists(image_path):
#                                 st.image(image_path, use_container_width=True)
#                             else:
#                                 st.warning(f"Img not found")
                            
#                             st.markdown(f"**{product.get('productDisplayName', 'N/A')}**")
#                             st.markdown(f"_{product.get('masterCategory', '')}_")
                            
#                             st.progress(int(product['similarity_score'] * 100))
#                             st.caption(f"Similarity: {product['similarity_score']:.2f}")

#     else:
#         st.info("Upload an image or enter a URL in the sidebar to start.")



import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import requests
from PIL import Image
import io
import math
from sklearn.neighbors import NearestNeighbors
import base64
import re
import inspect

# --- Page Configuration ---
st.set_page_config(
    page_title="Visual Product Matcher",
    page_icon="👕",
    layout="wide"
)

# --- Compatibility Wrapper for st.image ---
def safe_image(img, **kwargs):
    """Wrapper to handle use_column_width / use_container_width depending on Streamlit version."""
    sig = inspect.signature(st.image)
    if "use_container_width" in sig.parameters:
        st.image(img, use_container_width=True, **kwargs)
    else:
        st.image(img, use_column_width=True, **kwargs)

# --- Constants ---
ARTIFACTS_DIR = "."
CSV_FILE = "styles_deployment_sample.csv"
EMBEDDINGS_FILE = "emb_norm_sample.npy"
IMAGE_DIR = "images"

# --- Model & Data Loading ---
@st.cache_resource
def load_artifacts():
    try:
        extractor = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, "feature_extractor.keras"))

        # Load the sample CSV and construct the full image paths
        df = pd.read_csv(os.path.join(ARTIFACTS_DIR, CSV_FILE))
        df['image'] = df['id'].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg"))

        kmeans = joblib.load(os.path.join(ARTIFACTS_DIR, "kmeans.joblib"))
        emb_norm = np.load(os.path.join(ARTIFACTS_DIR, EMBEDDINGS_FILE))

        # Calculate centroids from the full kmeans model
        centroids = kmeans.cluster_centers_
        centroids_norm = centroids.astype("float32") / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)

        # Dynamically create cluster mappings from the loaded sample data
        unique_clusters = sorted(df['cluster'].unique())
        cluster_to_idx_map = {c: df[df['cluster'] == c].index.to_numpy() for c in unique_clusters}

        cluster_to_nn = {}
        for cluster_id, indices in cluster_to_idx_map.items():
            if len(indices) > 1:
                cluster_embeddings = emb_norm[indices]
                nn_model = NearestNeighbors(n_neighbors=min(20, len(indices)), metric="cosine", algorithm='brute')
                nn_model.fit(cluster_embeddings)
                cluster_to_nn[cluster_id] = nn_model
            else:
                cluster_to_nn[cluster_id] = None  # Handle small clusters

        return extractor, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn

    except FileNotFoundError as e:
        st.error(f"Error loading artifact: {e}. Please ensure all required files are in the GitHub repository.")
        st.stop()

# --- Helper Functions ---
def load_and_preprocess_image(image_source, source_type='file'):
    """Loads and preprocesses an image from file, URL, or base64 data URI."""
    try:
        if source_type == 'url':
            # --- Case 1: Base64 data URI ---
            if image_source.startswith("data:image"):
                match = re.match(r"data:image\/[a-zA-Z]+;base64,(.*)", image_source)
                if match:
                    image_data = base64.b64decode(match.group(1))
                    image = Image.open(io.BytesIO(image_data)).convert("RGB")
                else:
                    raise ValueError("Invalid base64 image string.")

            # --- Case 2: Google Drive link ---
            elif "drive.google.com" in image_source:
                file_id_match = re.search(r"/d/([a-zA-Z0-9_-]+)", image_source)
                if file_id_match:
                    file_id = file_id_match.group(1)
                    image_source = f"https://drive.google.com/uc?id={file_id}"
                else:
                    raise ValueError("Invalid Google Drive link format.")
                response = requests.get(image_source, stream=True, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")

            # --- Case 3: GitHub blob link ---
            elif "github.com" in image_source and not "raw.githubusercontent.com" in image_source:
                image_source = image_source.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                response = requests.get(image_source, stream=True, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")

            # --- Case 4: Standard HTTP(S) URL ---
            else:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(image_source, headers=headers, stream=True, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")

        else:  # File upload
            image = Image.open(image_source).convert("RGB")

        # Preprocess for model
        image_display = image.copy()
        image = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.resnet50.preprocess_input(img_array_expanded), image_display

    except Exception as e:
        st.error(f"Could not load image. Error: {e}")
        return None, None

def extract_embedding(image_tensor, extractor_model):
    embedding = extractor_model.predict(image_tensor, verbose=0)[0]
    embedding = embedding.astype("float32")
    embedding /= (np.linalg.norm(embedding) + 1e-10)
    return embedding

def search_similar(query_embedding, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn, top_k=12):
    dists_to_centroids = 1.0 - (centroids_norm @ query_embedding)
    closest_cluster = int(np.argmin(dists_to_centroids))

    nn_cluster = cluster_to_nn.get(closest_cluster)
    idx_pool = cluster_to_idx_map.get(closest_cluster, np.array([]))

    if nn_cluster is None or len(idx_pool) == 0:
        return pd.DataFrame()

    dist, ind = nn_cluster.kneighbors(query_embedding.reshape(1, -1), n_neighbors=min(top_k, len(idx_pool)))

    hits_indices = idx_pool[ind[0]]
    distances = dist[0]

    results_df = df.iloc[hits_indices].copy()
    results_df['cosine_distance'] = distances
    results_df['similarity_score'] = 1 - results_df['cosine_distance']
    return results_df.sort_values('cosine_distance')

# --- Main App Interface ---
st.title("👕 Visual Product Matcher")
st.write("Upload an image or provide a URL to find visually similar fashion products.")

if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'uploaded_image_display' not in st.session_state:
    st.session_state.uploaded_image_display = None

with st.spinner("Loading models and catalog..."):
    extractor, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn = load_artifacts()

st.sidebar.header("Search Options")
input_method = st.sidebar.radio("Input method:", ("Upload an Image", "Enter Image URL"))

query_image, uploaded_image_display = None, None
newly_uploaded = None

if input_method == "Upload an Image":
    uploaded_file = st.sidebar.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        newly_uploaded = uploaded_file.name
        query_image, uploaded_image_display = load_and_preprocess_image(uploaded_file, source_type='file')
else:
    url = st.sidebar.text_input("Enter image URL:")
    if url:
        newly_uploaded = url
        query_image, uploaded_image_display = load_and_preprocess_image(url, source_type='url')

if newly_uploaded != st.session_state.get('last_uploaded_name'):
    st.session_state.search_results = None
    st.session_state.last_uploaded_name = newly_uploaded

col1, col2 = st.columns([1, 3])

with col1:
    if uploaded_image_display:
        st.session_state.uploaded_image_display = uploaded_image_display
        st.header("Your Image")
        safe_image(uploaded_image_display)

        if st.button("Find Similar Products", type="primary"):
            with st.spinner("Searching for matches..."):
                query_embedding = extract_embedding(query_image, extractor)
                st.session_state.search_results = search_similar(query_embedding, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn)
    else:
        st.session_state.uploaded_image_display = None
        st.session_state.search_results = None

with col2:
    st.header("Similar Products")
    if st.session_state.search_results is not None:
        results = st.session_state.search_results

        min_similarity = st.slider("Filter by Similarity Score:", 0.0, 1.0, 0.0, 0.01)
        filtered_results = results[results['similarity_score'] >= min_similarity]

        if filtered_results.empty:
            st.warning("No results match the current filter. Try lowering the similarity score.")
        else:
            num_cols = 4
            num_rows = math.ceil(len(filtered_results) / num_cols)
            for i in range(num_rows):
                row_cols = st.columns(num_cols)
                for j in range(num_cols):
                    idx = i * num_cols + j
                    if idx < len(filtered_results):
                        product = filtered_results.iloc[idx]
                        with row_cols[j]:
                            image_path = product.get('image', '')
                            if os.path.exists(image_path):
                                safe_image(image_path)
                            else:
                                st.warning("Image not found")

                            st.markdown(f"**{product.get('productDisplayName', 'N/A')}**")
                            st.markdown(f"_{product.get('masterCategory', '')}_")

                            st.progress(int(product['similarity_score'] * 100))
                            st.caption(f"Similarity: {product['similarity_score']:.2f}")
    else:
        st.info("Upload an image or enter a URL in the sidebar to start.")
