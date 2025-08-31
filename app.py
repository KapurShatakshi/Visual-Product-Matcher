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

# --- Page Configuration (Mobile Responsive) ---
st.set_page_config(
    page_title="Visual Product Matcher",
    page_icon="👕",
    layout="wide"
)

# --- State Management Initialization ---
# This dictionary will now store all our persistent data
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'initial_results_df': None,
        'uploaded_image_display': None,
        'last_source_id': None
    }

# --- Artifact Loading ---
ARTIFACTS_DIR = "." 
IMAGE_DIR = "images"

@st.cache_resource
def load_artifacts():
    """Loads all necessary artifacts and dynamically builds cluster models."""
    try:
        extractor = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, "feature_extractor.keras"))
        df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "styles_deployment_sample.csv"))
        kmeans = joblib.load(os.path.join(ARTIFACTS_DIR, "kmeans.joblib"))
        emb_norm = np.load(os.path.join(ARTIFACTS_DIR, "emb_norm_sample.npy"))
        
        df['image_path'] = df['id'].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg"))
        
        # Dynamically create these to avoid errors
        centroids = kmeans.cluster_centers_
        centroids_norm = centroids.astype("float32") / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        
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
                cluster_to_nn[cluster_id] = None

        return extractor, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn

    except FileNotFoundError as e:
        st.error(f"Error loading artifact: {e}. Ensure all model files are in the directory.")
        st.stop()

# --- Helper Functions ---
def load_and_preprocess_image(image_source, source_type='file'):
    try:
        if source_type == 'url':
            response = requests.get(image_source, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        else: # file
            image = Image.open(image_source).convert("RGB")
        
        image_resized = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.resnet50.preprocess_input(img_array_expanded), image
    except Exception as e:
        st.error(f"Could not load image. Error: {e}")
        return None, None

def extract_embedding(image_tensor, extractor_model):
    embedding = extractor_model.predict(image_tensor, verbose=0)[0]
    embedding = embedding.astype("float32")
    embedding /= (np.linalg.norm(embedding) + 1e-10)
    return embedding

def search_similar(query_embedding, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn, top_k=12):
    """Finds the top_k visually similar products using NearestNeighbors."""
    dists_to_centroids = 1.0 - (centroids_norm @ query_embedding)
    closest_cluster = int(np.argmin(dists_to_centroids))
    
    nn_cluster = cluster_to_idx_map.get(closest_cluster)
    idx_pool = cluster_to_idx_map.get(closest_cluster, np.array([]))
    
    nn_model = cluster_to_nn.get(closest_cluster)

    if nn_model is None or len(idx_pool) == 0:
        return pd.DataFrame()

    dist, ind = nn_model.kneighbors(query_embedding.reshape(1, -1), n_neighbors=min(top_k, len(idx_pool)))
    
    hits_indices = idx_pool[ind[0]]
    distances = dist[0]
    
    results_df = df.iloc[hits_indices].copy()
    results_df['cosine_distance'] = distances
    results_df['similarity_score'] = 1 - results_df['cosine_distance']
    return results_df.sort_values('cosine_distance')

# --- Main App Interface ---
st.title("👕 Visual Product Matcher")
st.write("Upload an image or provide a URL to find visually similar fashion products.")

extractor, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn = load_artifacts()

st.sidebar.header("Search Options")
input_method = st.sidebar.radio("Input method:", ("Upload an Image", "Enter Image URL"))

uploaded_file = None
url = None

if input_method == "Upload an Image":
    uploaded_file = st.sidebar.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"])
else:
    url = st.sidebar.text_input("Enter image URL:")

# --- Logic to handle new image upload and perform search ONCE ---
source = uploaded_file if uploaded_file else url
if source:
    source_id = source.name if uploaded_file else source
    # Only re-process if the image has changed
    if st.session_state.app_state['last_source_id'] != source_id:
        st.session_state.app_state['last_source_id'] = source_id
        
        source_type = 'file' if uploaded_file else 'url'
        query_image_tensor, uploaded_image_display = load_and_preprocess_image(source, source_type)
        
        if query_image_tensor is not None:
            st.session_state.app_state['uploaded_image_display'] = uploaded_image_display
            with st.spinner("Analyzing image and finding top matches..."):
                query_embedding = extract_embedding(query_image_tensor, extractor)
                st.session_state.app_state['initial_results_df'] = search_similar(
                    query_embedding, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn
                )

# --- Layout and Display ---
col1, col2 = st.columns([1, 3])

with col1:
    if st.session_state.app_state['uploaded_image_display']:
        st.header("Your Image")
        st.image(st.session_state.app_state['uploaded_image_display'], use_container_width=True)

with col2:
    if st.session_state.app_state.get('initial_results_df') is not None:
        st.header("Similar Products")
        
        results_df = st.session_state.app_state['initial_results_df']
        
        if results_df.empty:
            st.warning("No similar products found. Please try another image.")
        else:
            # The slider now only filters the pre-fetched results
            min_similarity = st.slider("Filter results by similarity score:", 0.0, 1.0, 0.0, 0.01)
            
            filtered_results = results_df[results_df['similarity_score'] >= min_similarity]

            if filtered_results.empty:
                st.info("No products match the current filter. Slide down to see more results.")
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
                                image_path = product.get('image_path', '')
                                if os.path.exists(image_path):
                                    st.image(image_path, use_container_width=True)
                                else:
                                    st.warning(f"Img not found")
                                
                                st.markdown(f"**{product.get('productDisplayName', 'N/A')}**")
                                st.progress(int(product['similarity_score'] * 100))
                                st.caption(f"Similarity: {product['similarity_score']:.2f}")
    else:
        st.info("Upload an image or enter a URL in the sidebar to begin.")


