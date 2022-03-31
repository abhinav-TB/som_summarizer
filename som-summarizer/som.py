from sklearn_som.som import SOM
from sentence_transformers import SentenceTransformer
class som:
    def __init__(self,m,n,epochs):
        som = SOM(m, n, dim=768)
        som.fit(sentence_embeddings , epochs = 100)
    
    def generate_embeddings():
        sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        sentence_embeddings = sbert_model.encode(text_tokens)
