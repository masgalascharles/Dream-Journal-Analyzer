# Dream-Journal-Analyzer
I have been writing down my dreams every morning for the past months. I got the idea to analyze my dream journal to see frequent patterns over time. I used cosine similarity and agglomerative clustering, plus `llama-3.3-70b-versatile` to name the clusters.\
**FOR PRIVACY REASONS, I HAVE NOT INCLUDED MY PERSONAL DREAM JOURNAL IN THIS REPOSITORY. A SAMPLE DATASET HAS BEEN PROVIDED.**

## Dataset Format
dreams/\
├── 1.txt\
├── 2.txt\
├── 3.txt\
└── ...

Each file contains two lines:
```
YYYY-MM-DD
Dream entry
```

## How It Works
### 1. Generate embeddings
Each dream entry is converted into a 768-dimensional semantic embedding using the `all-mpnet-base-v2` SentenceTransformer model.
### 2. Similarities Matrix
Cosine similarity is calculated between each pair of dreams and stored in a matrix.
### 3. Clustering
The embeddings are normalized and clustered using agglomerative clustering.
### 4. Cluster titles
Titles for each cluster are generated using `llama-3.3-70b-versatile`.
### 5. Visuals
A scatter plot of the cluster assigned to each dream over time and a heatmap of the similarities matrix are generated.

## How to Run
1.
```cmd
git clone https://github.com/masgalascharles/Dream-Journal-Analyzer.git
```
2.
```cmd
pip install -r requirements.txt
```
3.
```cmd
set GROQ_API_KEY=your_key_here
```
4.
```cmd
python -m streamlit run app.py
```

## Screenshots
<img width="576" height="464" alt="image" src="https://github.com/user-attachments/assets/4416dc6d-99cc-4800-810d-6c0d6fcb7133" />
<img width="523" height="467" alt="image" src="https://github.com/user-attachments/assets/c796b491-e343-4eed-b9d9-ac6138af2fb9" />
<img width="522" height="365" alt="image" src="https://github.com/user-attachments/assets/3781f3cf-5a29-4d23-8045-486d6130a39d" />
