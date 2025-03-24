# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Step 1: Load and prepare the dataset
print("Loading dataset...")
data = pd.read_csv('complaints.csv')  # Adjust path to your downloaded file
data = data[['Consumer complaint narrative']].dropna()
data = data.sample(1000, random_state=42)  # Sample 1000 rows
print(f"Dataset loaded with {len(data)} samples.")

# Step 2: Preprocess the text with NLTK
def preprocess(text):
    """Preprocess text using NLTK: tokenize, remove stop words, lemmatize."""
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove stop words, punctuation, and lemmatize; keep meaningful words
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token.isalnum() and token not in stop_words]
    return " ".join(tokens)

print("Preprocessing text...")
data['processed_text'] = data['Consumer complaint narrative'].apply(preprocess)
print("Text preprocessing completed.")

# Step 3: Extract BERT embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    """Generate BERT embeddings for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

print("Generating BERT embeddings...")
embeddings_list = data['processed_text'].apply(get_bert_embeddings).tolist()
embeddings_array = np.array(embeddings_list)
print(f"Embeddings generated with shape: {embeddings_array.shape}")

# Step 4: Cluster with DBSCAN
print("Clustering with DBSCAN...")
similarity_matrix = cosine_similarity(embeddings_array)
distance_matrix = 1 - similarity_matrix
dbscan = DBSCAN(eps=0.3, min_samples=5, metric='precomputed')
clusters = dbscan.fit_predict(distance_matrix)
data['cluster'] = clusters
print("Clustering completed.")
print(data['cluster'].value_counts())

# Step 5: Visualize clusters with t-SNE
print("Visualizing clusters...")
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_array)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=data['cluster'], palette='viridis')
plt.title('DBSCAN Clusters of Customer Complaints')
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

if 0 in data['cluster'].values:
    print("\nTop words in cluster 0:")
    top_words = get_top_words(0)
    for word, freq in top_words:
        print(f"{word}: {freq}")
else:
    print("\nCluster 0 not found; check cluster labels.")
