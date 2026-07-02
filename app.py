import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
from datetime import datetime
from openai import OpenAI


################
## DATA SETUP ##
################

@st.cache_resource
def get_model():
    return SentenceTransformer("all-mpnet-base-v2")

model = get_model()

base_path = Path(__file__).parent
folder_path = base_path / "dreams"
data_length = len(list(Path(folder_path).glob("*.txt")))

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

@st.cache_data
def get_data(_model, folder_path, data_length):
    data = []
    texts = []
    
    for i in range(1, data_length + 1):
        file_path = folder_path / f"{i}.txt"

        if file_path.exists():
            lines = file_path.read_text().splitlines()

            data.append({
                "file_name": i,
                "date": datetime.strptime(lines[0], "%Y-%m-%d"),
                "text": lines[1]
            })
            texts.append(lines[1])
        else:
            print(f"{file_path} not found.")

    embeddings = _model.encode(texts)

    for i, embedding in enumerate(embeddings):
        data[i]["embedding"] = embedding

    data.sort(key=lambda x: x["file_name"])

    for i, entry in enumerate(data):
        entry["entry_index"] = i

    return data

data = get_data(model, folder_path, data_length)
file_names = np.array([entry["file_name"] for entry in data])
texts = np.array([entry["text"] for entry in data])
embeddings = np.array([entry["embedding"] for entry in data])
similarities_matrix = cosine_similarity(embeddings) # Get the cosine similarity between each pair of dreams


################
## CLUSTERING ##
################

n_clusters = st.slider("Number of Clusters", min_value=2, max_value=data_length // 2, value=data_length // 4)

normalized_embeddings = normalize(embeddings, norm="l2")
clusterer = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric="euclidean",
    linkage="ward"
)
cluster_assignment = clusterer.fit_predict(normalized_embeddings)
clusters = {}

for entry_index, cluster_index in enumerate(cluster_assignment):
    if cluster_index not in clusters:
        clusters[cluster_index] = []

    clusters[cluster_index].append(data[entry_index])

clusters = list(clusters.values())


# Sort by largest to smallest to easily see recurring patterns in dreams
clusters.sort(key=len, reverse=True)


# Easy access to a file's cluster ID
for i, cluster in enumerate(clusters):
    for entry in cluster:
        entry["cluster_id"] = i + 1
        data[entry["entry_index"]]["cluster_id"] = i + 1

cluster_ids = np.array([entry["cluster_id"] for entry in data])


# Cluster titles are generated using llama-3.3-70b-versatile
@st.cache_data
def generate_cluster_titles(n_clusters, clusters, _client):
    titles = []

    for i in range(0, len(clusters)):
        try:
            prompt = "I have the following dream journal entries grouped together in a data cluster:\n\n"

            for j in range(0, min(len(clusters[i]), 10)): # Only 10 entries are given to the LLM so this doesn't take forever when my dream journal grows
                prompt += f"{clusters[i][j]["text"]}\n\n"

            prompt += "Create a brief title for this cluster. Use the dominant themes/details/settings/etc. within this cluster to come up with your title. In your response, provide only the title with no additional text."
            response = _client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                frequency_penalty=0,
                presence_penalty=0
            )

            titles.append(response.choices[0].message.content.strip())
        except Exception as error:
            st.warning(f"Title generation failed: {error}")

            titles.append(f"Cluster #{i + 1}")

    return titles

cluster_titles = generate_cluster_titles(n_clusters, clusters, client)


# Calculate average similarity by getting the similarity between each dream in each cluster
def get_average_similarity(similarities_matrix, cluster):
    n = len(cluster)

    if n < 2:
        return 0
    
    total = 0
    
    for i in range(0, n):
        for j in range(i + 1, n):
            file1 = cluster[i]["entry_index"]
            file2 = cluster[j]["entry_index"]
            total += similarities_matrix[file1][file2]

    return total / (n * (n - 1) / 2)

average_similarities = [get_average_similarity(similarities_matrix, cluster) for cluster in clusters]


########
## UI ##
########

st.title("Dream Clusters")

cluster_selection = st.selectbox("Select Cluster", list(range(1, len(clusters) + 1)))

st.header(f"Cluster #{cluster_selection} - {cluster_titles[cluster_selection - 1]}")

cluster_index = cluster_selection - 1
selected_cluster = clusters[cluster_index]

st.write(f"Cluster size: {len(selected_cluster)}")
st.write(f"Average cosine similarity: {average_similarities[cluster_index]}")

for entry in selected_cluster:
    file_name = entry["file_name"]
    entry_index = entry["entry_index"]
    date = entry["date"]
    text = entry["text"]

    similarity_to_others = similarities_matrix[entry_index].copy()
    similarity_to_others[entry_index] = -np.inf
    highest_similarity_entry_indices = np.argpartition(similarity_to_others, -5)[-5:]
    highest_similarity_entry_indices = highest_similarity_entry_indices[np.argsort(similarity_to_others[highest_similarity_entry_indices])][::-1]
    similar_dreams_text = f"Five most similar dreams to Dream #{file_name}:\n\n"

    for i in highest_similarity_entry_indices:
        similar_entry = data[i]
        similarity_score = similarity_to_others[i]
        similar_dreams_text += f"Dream #{similar_entry["file_name"]} - {similar_entry["date"].strftime("%Y-%m-%d")}: {(np.round(similarity_score * 1000) / 1000):.3f}\n\n"

    with st.expander(f"Dream #{file_name} - {date.strftime("%Y-%m-%d")}"):
        st.write(text)
        st.divider()
        st.write(similar_dreams_text)

st.divider()


########################
## HEATMAP GENERATION ##
########################

st.title("Dream Similarity Heatmap")

def generate_heatmap(data, similarities_matrix):
    figure, axes = plt.subplots(figsize=(12, 10)) 
    image = axes.imshow(similarities_matrix, cmap="viridis", vmin=0, vmax=1)
    
    tick_labels = [entry["file_name"] for entry in data]
    
    axes.set_xticks(range(len(data)))
    axes.set_yticks(range(len(data)))

    axes.set_xticklabels(tick_labels, fontsize=7)
    axes.set_yticklabels(tick_labels, fontsize=7)
    
    plt.colorbar(image, ax=axes)
    axes.set_title(f"Dream Similarity Heatmap: {len(data)} Dreams", fontsize=15)
    
    return figure

st.pyplot(generate_heatmap(data, similarities_matrix))


############################
## SCATTER PLOT GENERATOR ##
############################

def generate_scatter_plot(data_length, file_names, cluster_ids, number_of_clusters):
    figure, axes = plt.subplots(figsize=(12, 8))

    axes.scatter(file_names, cluster_ids, color="black", s=100)
    axes.set_xticks(range(1, data_length + 1))
    axes.set_yticks(range(1, number_of_clusters + 1))
    axes.tick_params(axis="x", labelsize=7)
    axes.grid(axis="both", linestyle="--", alpha=0.6)
    axes.set_title("Dream Clusters Over Time", fontsize=16)
    axes.set_xlabel("File Name", fontsize=12)
    axes.set_ylabel("Cluster Number", fontsize=12)

    return figure

st.pyplot(generate_scatter_plot(data_length, file_names, cluster_ids, len(clusters)))
