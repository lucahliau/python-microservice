'''import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import signal

# Optional: catch SIGPIPE (useful in some environments)
def handle_sigpipe(signum, frame):
    sys.stderr.write("Received SIGPIPE. Exiting.\n")
    sys.stdout.flush()
    sys.exit(1)

signal.signal(signal.SIGPIPE, handle_sigpipe)

def get_products_dataframe(posts, model):
    """
    Convert the list of posts (from JSON) into a DataFrame.
    Each post should include at least an identifier and a description.
    """
    data = []
    for post in posts:
        try:
            # Use _id or id as the post identifier
            post_id = post.get('_id') or post.get('id')
            # Use 'description' if present, otherwise fall back to product_description
            description = post.get('description', post.get("product_description", ""))
            embedding = post.get('embedding', None)
            if embedding is None:
                embedding = model.encode(description).tolist()
            else:
                if isinstance(embedding, str):
                    try:
                        embedding = eval(embedding)
                    except Exception as e:
                        sys.stderr.write(f"Error converting embedding for post {post_id}: {e}\n")
                        embedding = model.encode(description).tolist()
            data.append({
                "id": post_id,
                "description": description,
                "embedding": np.array(embedding)
            })
        except Exception as e:
            sys.stderr.write(f"Error processing post {post}: {e}\n")
            continue
    try:
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        sys.stderr.write(f"Error creating DataFrame: {e}\n")
        raise

def recommend_products(user_liked_centers, user_disliked_centers, products_df, top_n=30, dislike_weight=1.0):
    """
    Compute a recommendation score for each product and return the top_n recommendations.
    """
    product_scores = []
    for idx, row in products_df.iterrows():
        try:
            product_embedding = row["embedding"].reshape(1, -1)
            liked_similarities = cosine_similarity(product_embedding, user_liked_centers)
            liked_score = liked_similarities.max()
            disliked_score = 0
            if user_disliked_centers is not None and len(user_disliked_centers) > 0:
                disliked_similarities = cosine_similarity(product_embedding, user_disliked_centers)
                disliked_score = disliked_similarities.max()
            final_score = liked_score - dislike_weight * disliked_score
            product_scores.append(final_score)
        except Exception as e:
            sys.stderr.write(f"Error scoring product at index {idx}: {e}\n")
            product_scores.append(-9999)
    products_df["final_score"] = product_scores
    recommended = products_df.sort_values("final_score", ascending=False)
    return recommended.head(top_n)

def get_recommendations(liked_clusters, disliked_clusters, posts):
    """
    Given liked/disliked clusters and a list of posts, compute and return recommendations.
    """
    try:
        user_liked_centers = np.array(liked_clusters)
        user_disliked_centers = np.array(disliked_clusters) if disliked_clusters else None
    except Exception as e:
        sys.stderr.write(f"Error converting clusters to numpy arrays: {e}\n")
        raise e
    
    sys.stderr.write("Initializing SentenceTransformer model...\n")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        sys.stderr.write(f"Error loading model: {e}\n")
        raise e
    sys.stderr.write("Model loaded successfully.\n")
    
    try:
        products_df = get_products_dataframe(posts, model)
    except Exception as e:
        sys.stderr.write(f"Error building products DataFrame: {e}\n")
        raise e
    
    try:
        recommendations = recommend_products(
            user_liked_centers,
            user_disliked_centers,
            products_df,
            top_n=30,
            dislike_weight=1.0
        )
    except Exception as e:
        sys.stderr.write(f"Error computing recommendations: {e}\n")
        raise e
    
    try:
        recommendations_list = recommendations[['id', 'description', 'final_score']].to_dict(orient='records')
        return recommendations_list
    except Exception as e:
        sys.stderr.write(f"Error preparing output: {e}\n")
        raise e

# Optional: For local testing via command line.
if __name__ == "__main__":
    input_data = sys.stdin.read()
    if not input_data:
        sys.stderr.write("No input data received.\n")
        print(json.dumps({"error": "No input data received"}))
        sys.stdout.flush()
        exit(1)
    try:
        data = json.loads(input_data)
    except Exception as e:
        sys.stderr.write(f"Error parsing JSON: {e}\n")
        print(json.dumps({"error": "Error parsing JSON", "details": str(e)}))
        sys.stdout.flush()
        exit(1)
    
    liked_clusters = data.get("likedClusters", [])
    disliked_clusters = data.get("dislikedClusters", [])
    posts = data.get("posts", [])
    
    if not liked_clusters or not posts:
        msg = "Missing required data: likedClusters and posts are required."
        sys.stderr.write(msg + "\n")
        print(json.dumps({"error": msg}))
        sys.stdout.flush()
        exit(1)
    
    recommendations_list = get_recommendations(liked_clusters, disliked_clusters, posts)
    print(json.dumps(recommendations_list))
    sys.stdout.flush()
    '''
import sys
import time
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import signal

# Optional: catch SIGPIPE (useful in some environments)
def handle_sigpipe(signum, frame):
    sys.stderr.write("Received SIGPIPE. Exiting.\n")
    sys.stdout.flush()
    sys.exit(1)

signal.signal(signal.SIGPIPE, handle_sigpipe)

def get_products_dataframe(posts, model):
    """
    Convert the list of posts (from JSON) into a DataFrame.
    Each post should include at least an identifier and a description.
    """
    data = []
    for post in posts:
        try:
            # Use _id or id as the post identifier
            post_id = post.get('_id') or post.get('id')
            # Use 'description' if present, otherwise fall back to product_description
            description = post.get('description', post.get("product_description", ""))
            embedding = post.get('embedding', None)
            if embedding is None:
                embedding = model.encode(description).tolist()
            else:
                if isinstance(embedding, str):
                    try:
                        embedding = eval(embedding)
                    except Exception as e:
                        sys.stderr.write(f"Error converting embedding for post {post_id}: {e}\n")
                        embedding = model.encode(description).tolist()
            data.append({
                "id": post_id,
                "description": description,
                "embedding": np.array(embedding)
            })
        except Exception as e:
            sys.stderr.write(f"Error processing post {post}: {e}\n")
            continue
    try:
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        sys.stderr.write(f"Error creating DataFrame: {e}\n")
        raise

def recommend_products(user_liked_centers, user_disliked_centers, products_df, top_n=30, dislike_weight=1.0):
    """
    Compute a recommendation score for each product and return the top_n recommendations.
    """
    product_scores = []
    for idx, row in products_df.iterrows():
        try:
            product_embedding = row["embedding"].reshape(1, -1)
            liked_similarities = cosine_similarity(product_embedding, user_liked_centers)
            liked_score = liked_similarities.max()
            disliked_score = 0
            if user_disliked_centers is not None and len(user_disliked_centers) > 0:
                disliked_similarities = cosine_similarity(product_embedding, user_disliked_centers)
                disliked_score = disliked_similarities.max()
            final_score = liked_score - dislike_weight * disliked_score
            product_scores.append(final_score)
        except Exception as e:
            sys.stderr.write(f"Error scoring product at index {idx}: {e}\n")
            product_scores.append(-9999)
    products_df["final_score"] = product_scores
    recommended = products_df.sort_values("final_score", ascending=False)
    return recommended.head(top_n)

def get_recommendations(liked_clusters, disliked_clusters, posts):
    """
    Given liked/disliked clusters and a list of posts, compute and return recommendations.
    """
    try:
        user_liked_centers = np.array(liked_clusters)
        user_disliked_centers = np.array(disliked_clusters) if disliked_clusters else None
    except Exception as e:
        sys.stderr.write(f"Error converting clusters to numpy arrays: {e}\n")
        raise e
    
    sys.stderr.write("Initializing SentenceTransformer model...\n")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        sys.stderr.write(f"Error loading model: {e}\n")
        raise e
    sys.stderr.write("Model loaded successfully.\n")
    
    try:
        products_df = get_products_dataframe(posts, model)
    except Exception as e:
        sys.stderr.write(f"Error building products DataFrame: {e}\n")
        raise e
    
    try:
        recommendations = recommend_products(
            user_liked_centers,
            user_disliked_centers,
            products_df,
            top_n=30,
            dislike_weight=1.0
        )
    except Exception as e:
        sys.stderr.write(f"Error computing recommendations: {e}\n")
        raise e
    
    try:
        recommendations_list = recommendations[['id', 'description', 'final_score']].to_dict(orient='records')
        return recommendations_list
    except Exception as e:
        sys.stderr.write(f"Error preparing output: {e}\n")
        raise e

# Optional: For local testing via command line.
if __name__ == "__main__":
    start_time = time.perf_counter()  # Start the timer
    
    input_data = sys.stdin.read()
    if not input_data:
        sys.stderr.write("No input data received.\n")
        print(json.dumps({"error": "No input data received"}))
        sys.stdout.flush()
        exit(1)
    try:
        data = json.loads(input_data)
    except Exception as e:
        sys.stderr.write(f"Error parsing JSON: {e}\n")
        print(json.dumps({"error": "Error parsing JSON", "details": str(e)}))
        sys.stdout.flush()
        exit(1)
    
    liked_clusters = data.get("likedClusters", [])
    disliked_clusters = data.get("dislikedClusters", [])
    posts = data.get("posts", [])
    
    if not liked_clusters or not posts:
        msg = "Missing required data: likedClusters and posts are required."
        sys.stderr.write(msg + "\n")
        print(json.dumps({"error": msg}))
        sys.stdout.flush()
        exit(1)
    
    recommendations_list = get_recommendations(liked_clusters, disliked_clusters, posts)
    
    # Log each recommended post's score.
    sys.stderr.write("Recommended Posts and their Scores:\n")
    for rec in recommendations_list:
        sys.stderr.write(f"Post ID: {rec['id']} - Score: {rec['final_score']}\n")
    sys.stderr.flush()
    
    print(json.dumps(recommendations_list))
    sys.stdout.flush()
    
    end_time = time.perf_counter()  # End the timer
    sys.stderr.write(f"Script completed in {end_time - start_time:.2f} seconds.\n")
    sys.stderr.flush()


