from sklearn_som.som import SOM
from sentence_transformers import SentenceTransformer
class som:
    def __init__(self,m,n,epochs,text_tokens):
        self.som_obj = SOM(m, n, dim=768)
        self.text_tokens = text_tokens

    def predict(self):
        sentence_embeddings = self.generate_embeddings()
        self.som_obj.fit(sentence_embeddings , epochs = 100)
        return self.som_obj.predict(sentence_embeddings)
    def generate_embeddings(self):
        sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        return sbert_model.encode(self.text_tokens)