from sklearn_som.som import SOM
import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

class som:
    def __init__(self,m,n,epochs,text_tokens):
        self.som_obj = SOM(m, n, dim=768)
        self.text_tokens = text_tokens
        self.epochs = epochs

    def predict(self):
        sentence_embeddings = self.generate_embeddings()
        self.som_obj.fit(sentence_embeddings , self.epochs)
        return self.som_obj.predict(sentence_embeddings)

    def generate_embeddings(self):
        sbert_model = load_model()
        return sbert_model.encode(self.text_tokens)
