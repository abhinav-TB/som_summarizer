import string
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
# nltk.download('wordnet')

class preprocessing:
    def __init__(self):
        self.text = None
        self.text_tokens = None
    
    def __call__(self , text:string) -> None:
        self.text = text 
        self.tokensize()
        self.remove_punctuation()
        self.remove_stopwords()
        self.lemmatize()
        return self.text_tokens

    def tokensize(self):
        self.text_tokens = sent_tokenize(self.text)

    def remove_punctuation(self):
        for i in range(len(self.text_tokens)):
            self.text_tokens[i] = "".join([i for  i in self.text_tokens[i] if i not in string.punctuation])
    
    def remove_stopwords(self):
        stop_words = set(stopwords.words('english'))

        new_text_tokens = []

        for sentence in self.text_tokens:
            temp = " ".join([w for w in sentence.split() if not w.lower() in stop_words])
            new_text_tokens.append(temp)

        self.text_tokens = new_text_tokens
    
    def lemmatize(self):
        lemmatizer = WordNetLemmatizer()

        new_text_tokens = []

        for sentence in self.text_tokens:
            temp = " ".join([lemmatizer.lemmatize(w) for w in sentence.split()])
            new_text_tokens.append(temp)

        self.text_tokens = new_text_tokens