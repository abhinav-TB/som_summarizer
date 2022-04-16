from .preprocessing import preprocessor
from .features import features
from .som import som
import primefac
class summarizer:
    """
    abstracts the entire summarization procedure 
    """
    def __init__(self,epochs,sum_size) -> None:
        """
        initializes the summarizer

        Parameters
        ----------
        epochs : int
            number of epochs for the som
        m : int
            number of rows in the som
        n : int
            number of columns in the som
        """
        self.input = None
        self.org_tokens = None 
        self.pos_mapping = {}
        self.org_mapping = {}   
        self.epochs = epochs
        self.text_tokens = None
        self.sum_size = sum_size

    
    def generate_summary(self,input):
        """
        generates the summary

        Parameters
        ----------
        input : str
            input text to be summarized
        
        Returns
        -------
        str
            summary of the input text
        """
        self.input = input
        self.pre_process()# text_tokens mapping to org sentence
        self.create_mappings()
        f = features(self.text_tokens)
        self.scores = f.score()
        factors = list( primefac.primefac(self.sum_size) )
        factors.sort(reverse=True)
        if not factors:
            self.m = 1
            self.n = 1
        else:
            self.m = factors[0]
            self.n = int(self.sum_size/factors[0])
    
        s = som(self.m,self.n,self.epochs,self.text_tokens)
        self.predictions = s.predict()
        cluster_len = self.m*self.n
        clusters = [[] for _ in range(cluster_len)]
        for i , sentence in enumerate(self.text_tokens):
            clusters[self.predictions[i]].append(sentence)
        
        summary = []
        for cluster in clusters:
            summary.append(self.org_mapping[sorted(cluster,key=lambda x:self.scores[x],reverse = True)[0]])
        summary = sorted(summary,key = lambda x:self.pos_mapping[x])  # sorting to preserve order
        self.summary = "".join(summary)
        return self.summary

    def create_mappings(self):
        """
        creates the mappings between the original sentence and the text tokens
        """
        for i ,sentence in enumerate(self.text_tokens):
            self.org_mapping[sentence] = self.org_tokens[i]

        for i , sentence in enumerate(self.org_tokens):
            self.pos_mapping[sentence] = i

    def pre_process(self):
        """
        preprocesses the input text
        """
        preprocesor = preprocessor()
        self.text_tokens ,self.org_tokens = preprocesor(self.input)
        for i, token in enumerate(self.text_tokens):
            if token == "":
                self.text_tokens.pop(i)
                self.org_tokens.pop(i)