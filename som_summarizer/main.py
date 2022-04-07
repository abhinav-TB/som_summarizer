from .preprocessing import preprocessor
from .features import features
from .som import som
from math import ceil
import primefac
class summarizer:
    def __init__(self,epochs,sumSz) -> None:
        self.input = None
        self.org_tokens = None 
        self.pos_mapping = {}
        self.org_mapping = {}   
        self.epochs = epochs
        self.text_tokens = None
        factors = list( primefac.primefac(sumSz) )
        factors.sort(reverse=True)
        if not factors:
            self.m = 1
            self.n = 1
        else:
            self.m = factors[0]
            self.n = int(sumSz/factors[0])

    
    def generate_summary(self,input):
        self.input = input
        self.pre_process()# text_tokens mapping to org sentence
        self.create_mappings()
        f = features(self.text_tokens)
        self.scores = f.score()
        s = som(self.m,self.n,self.epochs,self.text_tokens)
        self.predictions = s.predict()
        cluster_len = max(self.predictions) + 1
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
        for i ,sentence in enumerate(self.text_tokens):
            self.org_mapping[sentence] = self.org_tokens[i]

        for i , sentence in enumerate(self.org_tokens):
            self.pos_mapping[sentence] = i

    def pre_process(self):
        preprocesor = preprocessor()
        self.text_tokens ,self.org_tokens = preprocesor(self.input)
        for i, token in enumerate(self.text_tokens):
            if token == "":
                self.text_tokens.pop(i)
                self.org_tokens.pop(i)

        

