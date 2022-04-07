import string
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
# nltk.download('wordnet')

class preprocessor:
    """
    preprocess the input text  
    
    """
    def __init__(self):
        """
        declare the class variables
        """
        self.text = None
        self.text_tokens = None
        self.org_tokens = None
    
    def __call__(self , text:string) -> None:
        """
        preprocess the input text

        parameters:
        text: input text

        returns:
        text_tokens: list of text tokens
        """
        
        self.text = text 
        self.tokensize()
        self.remove_punctuation()
        self.remove_stopwords()
        self.lemmatize()
        return self.text_tokens , self.org_tokens

    def tokensize(self):
        """
        tokenize the input text

        """
        self.text_tokens = sent_tokenize(self.text)
        self.org_tokens = self.text_tokens.copy()

    def remove_punctuation(self):
        """
        remove punctuation from the input text

        """
        for i in range(len(self.text_tokens)):
            self.text_tokens[i] = "".join([i for  i in self.text_tokens[i] if i not in string.punctuation])
    
    def remove_stopwords(self):
        """
        remove stopwords from the input text

        """
        stop_words = set(stopwords.words('english'))

        new_text_tokens = []

        for sentence in self.text_tokens:
            temp = " ".join([w for w in sentence.split() if not w.lower() in stop_words])
            new_text_tokens.append(temp)

        self.text_tokens = new_text_tokens
    
    def lemmatize(self):
        """
        lemmatize the input text
        
        """
        lemmatizer = WordNetLemmatizer()

        new_text_tokens = []

        for sentence in self.text_tokens:
            temp = " ".join([lemmatizer.lemmatize(w) for w in sentence.split()])
            new_text_tokens.append(temp)

        self.text_tokens = new_text_tokens