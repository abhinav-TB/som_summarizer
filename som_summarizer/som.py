from sklearn_som.som import SOM
from sentence_transformers import SentenceTransformer
class som:
    """
    train the input text on the SOM
    """
    def __init__(self,m,n,epochs,text_tokens):
        """
        initialize the SOM

        parameters:
        m: number of rows
        n: number of columns
        epochs: number of epochs
        text_tokens: list of text tokens

        """
        self.som_obj = SOM(m, n, dim=768)
        self.text_tokens = text_tokens
        self.epochs = epochs

    def predict(self):
        """
        Predict the clusters of the input text

        parameters:
        none

        returns:
        clusters: list of clusters
        """
        sentence_embeddings = self.generate_embeddings()
        self.som_obj.fit(sentence_embeddings , self.epochs)
        return self.som_obj.predict(sentence_embeddings)
    def generate_embeddings(self):
        """
        generate the sentence embeddings of the input text

        parameters:
        none

        returns:
        sentence_embeddings: list of sentence embeddings
        """
        sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        return sbert_model.encode(self.text_tokens)


