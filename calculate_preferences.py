'''import sys
import json
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Global model variable
model = None

def get_model(model_name="all-MiniLM-L6-v2"):
    global model
    if model is None:
        try:
            sys.stderr.write("Loading SentenceTransformer model...\n")
            sys.stderr.flush()
            model = SentenceTransformer(model_name)
        except Exception as e:
            sys.stderr.write(f"Error loading model: {e}\n")
            sys.stderr.flush()
            raise
    return model

def cluster_descriptions(descriptions, num_clusters=None, model_name="all-MiniLM-L6-v2"):
    if not descriptions:
        return []
    try:
        model_instance = get_model(model_name)
    except Exception as e:
        sys.stderr.write(f"Error obtaining model: {e}\n")
        sys.stderr.flush()
        raise
    sys.stderr.write(f"Clustering {len(descriptions)} descriptions\n")
    sys.stderr.flush()
    try:
        embeddings = model_instance.encode(descriptions)
    except Exception as e:
        sys.stderr.write(f"Error encoding descriptions: {e}\n")
        sys.stderr.flush()
        raise
    if num_clusters is None:
        num_clusters = min(3, len(embeddings))
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(embeddings)
        centers = kmeans.cluster_centers_.tolist()
    except Exception as e:
        sys.stderr.write(f"Error during clustering: {e}\n")
        sys.stderr.flush()
        raise
    return centers

def calculate_preferences(liked_descriptions, disliked_descriptions):
    """
    Given liked and disliked descriptions, compute cluster centers for each.
    Returns a dict with likedClusters and dislikedClusters.
    """
    try:
        liked_clusters = cluster_descriptions(liked_descriptions) if liked_descriptions else []
    except Exception as e:
        sys.stderr.write(f"Error clustering liked descriptions: {e}\n")
        liked_clusters = []
    try:
        disliked_clusters = cluster_descriptions(disliked_descriptions) if disliked_descriptions else []
    except Exception as e:
        sys.stderr.write(f"Error clustering disliked descriptions: {e}\n")
        disliked_clusters = []
    output = {"likedClusters": liked_clusters, "dislikedClusters": disliked_clusters}
    sys.stderr.write(f"Output clusters: {output}\n")
    sys.stderr.flush()
    return output

# Optional: You can keep a main() for local testing if needed.
if __name__ == "__main__":
    
    # Read JSON input from stdin for local testing.
    input_data = sys.stdin.read()
    data = json.loads(input_data)
    liked = data.get("likedDescriptions", [])
    disliked = data.get("dislikedDescriptions", [])
    result = calculate_preferences(liked, disliked)
    print(json.dumps(result))
    sys.stdout.flush()
'''
import sys
import time
import json
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Global model variable
model = None

def get_model(model_name="all-MiniLM-L6-v2"):
    global model
    if model is None:
        try:
            sys.stderr.write("Loading SentenceTransformer model...\n")
            sys.stderr.flush()
            model = SentenceTransformer(model_name)
        except Exception as e:
            sys.stderr.write(f"Error loading model: {e}\n")
            sys.stderr.flush()
            raise
    return model

def cluster_descriptions(descriptions, num_clusters=None, model_name="all-MiniLM-L6-v2"):
    if not descriptions:
        return []
    try:
        model_instance = get_model(model_name)
    except Exception as e:
        sys.stderr.write(f"Error obtaining model: {e}\n")
        sys.stderr.flush()
        raise
    sys.stderr.write(f"Clustering {len(descriptions)} descriptions\n")
    sys.stderr.flush()
    try:
        embeddings = model_instance.encode(descriptions)
    except Exception as e:
        sys.stderr.write(f"Error encoding descriptions: {e}\n")
        sys.stderr.flush()
        raise
    if num_clusters is None:
        num_clusters = min(3, len(embeddings))
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(embeddings)
        centers = kmeans.cluster_centers_.tolist()
    except Exception as e:
        sys.stderr.write(f"Error during clustering: {e}\n")
        sys.stderr.flush()
        raise
    return centers

def calculate_preferences(liked_descriptions, disliked_descriptions):
    """
    Given liked and disliked descriptions, compute cluster centers for each.
    Returns a dict with likedClusters and dislikedClusters.
    """
    try:
        liked_clusters = cluster_descriptions(liked_descriptions) if liked_descriptions else []
    except Exception as e:
        sys.stderr.write(f"Error clustering liked descriptions: {e}\n")
        liked_clusters = []
    try:
        disliked_clusters = cluster_descriptions(disliked_descriptions) if disliked_descriptions else []
    except Exception as e:
        sys.stderr.write(f"Error clustering disliked descriptions: {e}\n")
        disliked_clusters = []
    output = {"likedClusters": liked_clusters, "dislikedClusters": disliked_clusters}
    sys.stderr.write(f"Output clusters: {output}\n")
    sys.stderr.flush()
    return output

# Optional: You can keep a main() for local testing if needed.
if __name__ == "__main__":
    start_time = time.perf_counter()  # Start the timer

    # Read JSON input from stdin for local testing.
    input_data = sys.stdin.read()
    data = json.loads(input_data)
    liked = data.get("likedDescriptions", [])
    disliked = data.get("dislikedDescriptions", [])
    result = calculate_preferences(liked, disliked)
    print(json.dumps(result))
    sys.stdout.flush()

    end_time = time.perf_counter()  # End the timer
    sys.stderr.write(f"Script completed in {end_time - start_time:.2f} seconds.\n")
    sys.stderr.flush()

