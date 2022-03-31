from .utils.tfidf import tf_idf
import math
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
nltk.download('averaged_perceptron_tagger')
class features:
    def __init__(self,text_tokens):
        self.text_tokens = text_tokens

    def score(self):
        tf_idf_score = self.tfidf()
        sentence_position_score = self.sentence_position()
        noun_numerical_count_score = self.noun_numerical_count()
        return {**tf_idf_score, **sentence_position_score, **noun_numerical_count_score}

    def tfidf(self):
        tfidf = tf_idf(self.text_tokens)
        return tfidf.score()

    def sentence_position(self):
        tt = dict()
        thresh = len(self.text_tokens) * 0.2
        min = thresh * len(self.text_tokens)
        max = thresh * 2 * len(self.text_tokens)
        for i in range(0,len(self.text_tokens)):
            if ( i == 0 or i == (len(self.text_tokens) - 1) ):
                tt[self.text_tokens[i]] = 2
            else:
                tt[self.text_tokens[i]] = float(math.cos((i - min) / ((1 / max) - min )))
        
        return tt

    def noun_numerical_count(self):
        # NOUN AND NUMERAL COUNTS
        sentence = ''
        # initializing weights for nouns and numerals
        nounwt = 0.3 
        numwt = 0.7

        for t in self.text_tokens:
            sentence = sentence + t + '. '

            tokens = sent_tokenize(sentence)

            noun_numerical_count = dict()
        for t in tokens:
            tok2 = nltk.word_tokenize(t)
            t2 = nltk.Text(tok2)
            tags = nltk.pos_tag(t2)
        counts = Counter(tag for word, tag in tags if tag == 'NN' or tag == 'CD')
        if 'NN' in counts and 'CD' in counts:
            noun_numerical_count[t[:-1]] = counts['NN']*nounwt + counts['CD']*numwt
        elif 'NN' in counts and 'CD' not in counts:
            noun_numerical_count[t[:-1]] = counts['NN']*nounwt
        elif 'NN' not in counts and 'CD' in counts:
            noun_numerical_count[t[:-1]] = counts['CD']*numwt
        else:
            noun_numerical_count[t[:-1]] = 0
        
        return noun_numerical_count