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

# --- Artifact Loading from Local Project Files ---
ARTIFACTS_DIR = "." 
IMAGE_DIR = "images" # Define the image directory

@st.cache_resource
def load_artifacts():
    """Loads all necessary artifacts and dynamically builds cluster models."""
    try:
        # Load the core models and data files from your project directory
        extractor = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, "feature_extractor.keras"))
        df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "styles_deployment_sample.csv"))
        kmeans = joblib.load(os.path.join(ARTIFACTS_DIR, "kmeans.joblib"))
        emb_norm = np.load(os.path.join(ARTIFACTS_DIR, "emb_norm_sample.npy"))
        
        # FIX: Create the correct image path, matching the notebook's logic
        df['image_path'] = df['id'].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg"))
        
        # Calculate centroids directly from the loaded kmeans model
        centroids = kmeans.cluster_centers_
        centroids_norm = centroids.astype("float32") / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        
        # FIX for IndexError: Dynamically create cluster mappings from the loaded sample data
        unique_clusters = sorted(df['cluster'].unique())
        cluster_to_idx_map = {c: df[df['cluster'] == c].index.to_numpy() for c in unique_clusters}

        # Recreate the per-cluster Nearest Neighbor models in memory
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
        # Basic error handling for missing files
        st.error(f"Error loading artifact: {e}. Please ensure all model and data files are in the main project directory.")
        st.stop()

# --- Helper Functions ---
def load_and_preprocess_image(image_source, source_type='file'):
    """Loads and preprocesses an image from a file or URL with error handling."""
    try:
        if source_type == 'url':
            response = requests.get(image_source, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        else: # file
            image = Image.open(image_source).convert("RGB")
        
        image = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.resnet50.preprocess_input(img_array_expanded), image
    except Exception as e:
        st.error(f"Could not load image. Error: {e}")
        return None, None

def extract_embedding(image_tensor, extractor_model):
    """Extracts and normalizes the feature embedding from an image."""
    embedding = extractor_model.predict(image_tensor, verbose=0)[0]
    embedding = embedding.astype("float32")
    embedding /= (np.linalg.norm(embedding) + 1e-10)
    return embedding

def search_similar(query_embedding, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn, top_k=12):
    """Finds visually similar products using the notebook's two-step search logic."""
    dists_to_centroids = 1.0 - (centroids_norm @ query_embedding)
    closest_cluster = int(np.argmin(dists_to_centroids))
    
    nn_cluster = cluster_to_nn.get(closest_cluster)
    idx_pool = cluster_to_idx_map.get(closest_cluster, np.array([]))

    if nn_cluster is None or len(idx_pool) == 0:
        st.warning("Could not find a matching product cluster. Please try a different image.")
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

with st.spinner("Loading models and product catalog..."):
    extractor, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn = load_artifacts()

st.sidebar.header("Search Options")
input_method = st.sidebar.radio("Input method:", ("Upload an Image", "Enter Image URL"))

query_image, uploaded_image_display = None, None

if input_method == "Upload an Image":
    uploaded_file = st.sidebar.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        query_image, uploaded_image_display = load_and_preprocess_image(uploaded_file, source_type='file')
else:
    url = st.sidebar.text_input("Enter image URL:")
    if url:
        query_image, uploaded_image_display = load_and_preprocess_image(url, source_type='url')
        
col1, col2 = st.columns([1, 3])

if uploaded_image_display:
    with col1:
        st.header("Your Image")
        st.image(uploaded_image_display, use_column_width=True) # <-- FIX: Changed to use_column_width
        search_button = st.button("Find Similar Products", type="primary")
else:
    search_button = False

with col2:
    if search_button and query_image is not None:
        st.header("Similar Products")
        with st.spinner("Searching for matches..."):
            query_embedding = extract_embedding(query_image, extractor)
            results = search_similar(query_embedding, df, kmeans, centroids_norm, cluster_to_idx_map, cluster_to_nn)
            
            min_similarity = st.slider("Filter by Similarity Score:", 0.0, 1.0, 0.5, 0.01)
            max_dist_filter = 1 - min_similarity
            filtered_results = results[results['cosine_distance'] <= max_dist_filter]

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
                                image_path = product.get('image_path', '')
                                if os.path.exists(image_path):
                                    st.image(image_path, use_column_width=True) # <-- FIX: Changed to use_column_width
                                else:
                                    st.warning(f"Img not found: {image_path}")
                                
                                st.markdown(f"**{product.get('productDisplayName', 'N/A')}**")
                                st.markdown(f"_{product.get('masterCategory', '')} / {product.get('articleType', '')}_")
                                
                                st.progress(int(product['similarity_score'] * 100))
                                st.caption(f"Similarity: {product['similarity_score']:.2f}")

    elif not uploaded_image_display:
        st.info("Upload an image or enter a URL in the sidebar to start.")

