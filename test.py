#Import necessary libraries
import pandas as pd
import numpy as np
import nltk
nltk.data.path.append("./nltk_data")  # Adjust if needed
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Step 1: Load and prepare the dataset
print("Loading dataset...")
data = pd.read_excel('input_file.xlsx')  # Your file
data = data[['rca']].dropna()
# data = data.sample(5000, random_state=42)  # Uncomment if sampling is needed
print(f"Dataset loaded with {len(data)} samples.")

# Step 2: Preprocess the text with NLTK
def preprocess(text):
    """Preprocess text using NLTK: tokenize, remove stop words, lemmatize."""
    tokens = word_tokenize(str(text).lower())  # Convert to string to handle non-string inputs
    tokens = [lemmatizer.lemmatize(token) for token in tokens
              if token.isalnum() and token not in stop_words]
    return " ".join(tokens)

print("Preprocessing text...")
data['processed_text'] = data['rca'].apply(preprocess)
print("Text preprocessing completed.")
print(data['processed_text'].head())  # Preview first few rows

# Step 3: Extract domain-based embeddings with Sentence Transformers
print("Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, domain-agnostic but effective

def get_embeddings(texts):
    """Generate embeddings using Sentence Transformers."""
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

print("Generating embeddings...")
embeddings_array = get_embeddings(data['processed_text'].tolist())
print(f"Embeddings generated with shape: {embeddings_array.shape}")  # Should be (n_samples, 384)

# Step 4: Cluster with HDBSCAN
print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,       # Minimum size of a cluster
    metric='euclidean',       # Use Euclidean distance (common with Sentence Transformers)
    cluster_selection_method='eom'  # Excess of Mass for cluster selection
)
clusters = clusterer.fit_predict(embeddings_array)
data['cluster'] = clusters
print("Clustering completed.")
print(data['cluster'].value_counts())  # -1 indicates noise

# Step 5: Visualize clusters with t-SNE
print("Visualizing clusters...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)  # Adjust perplexity if needed
embeddings_2d = tsne.fit_transform(embeddings_array)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=data['cluster'], palette='viridis')
plt.title('HDBSCAN Clusters of Customer Complaints')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

# Step 6: Interpret clusters by finding top words
def get_top_words(cluster_id, n_words=10):
    """Get the most frequent words in a given cluster."""
    cluster_texts = data[data['cluster'] == cluster_id]['processed_text']
    all_words = " ".join(cluster_texts).split()
    word_freq = Counter(all_words)
    return word_freq.most_common(n_words)

# Check and print top words for a few clusters
for cluster_id in sorted(set(clusters) - {-1}):  # Exclude noise (-1)
    print(f"\nTop words in cluster {cluster_id}:")
    top_words = get_top_words(cluster_id)
    for word, freq in top_words:
        print(f"{word}: {freq}")
if -1 in clusters:
    print("\nNoise points detected (-1 label).")
