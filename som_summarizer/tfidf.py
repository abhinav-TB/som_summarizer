import math
from nltk import  word_tokenize

class tf_idf:
    """
    calculate the tfidf score of a sentence
    """
    def __init__(self,text_tokens):
        """
        initializing the class with the text tokens
        """
        self.text_tokens = text_tokens
        self.tfid = {}

    def score(self) -> dict:
        """
        score a sentence by its word's Term frequency Inverse Document frequency
        Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.

        parameters:
        none

        returns:
        score of each sentence
        """
        
        total_documents = len(self.text_tokens)
        freq_matrix = self.create_frequency_matrix(self.text_tokens)

        tf_matrix = self.create_tf_matrix(freq_matrix)

        count_doc_per_words = self.create_documents_per_words(freq_matrix)

        idf_matrix = self.create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)

        tf_idf_matrix = self.create_tf_idf_matrix(tf_matrix, idf_matrix)

        return  self.score_sentences(tf_idf_matrix)
    

    def create_frequency_matrix(self,sentences):
        """
        accepting sentences as inputs
        creating a mapping between a sentence and freq of words in that sentence
        """
        frequency_matrix = {}
        for sent in sentences:
            freq_table = {}
            words = word_tokenize(sent)
            for word in words:

                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1

            frequency_matrix[sent] = freq_table

        return frequency_matrix

    def create_tf_matrix(self,freq_matrix):
        """
        term frequency matrix - the number of times a word occur in a sentence / total number of words in that sentence
        """
        tf_matrix = {}

        for sent, f_table in freq_matrix.items():
            tf_table = {}

            count_words_in_sentence = len(f_table)
            for word, count in f_table.items():
                tf_table[word] = count / count_words_in_sentence

            tf_matrix[sent] = tf_table

        return tf_matrix

    def create_documents_per_words(self,freq_matrix):
        """
        finds freqency of the words in the entire corpus
        """
        word_per_doc_table = {}

        for sent, f_table in freq_matrix.items():
            for word, count in f_table.items():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1

        return word_per_doc_table
    
    def create_idf_matrix(self,freq_matrix, count_doc_per_words, total_documents):
        """
        creating the idf matrix with the direct log formula
        """
        idf_matrix = {}

        for sent, f_table in freq_matrix.items():
            idf_table = {}

            for word in f_table.keys():
                idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

            idf_matrix[sent] = idf_table

        return idf_matrix

    def create_tf_idf_matrix(self,tf_matrix, idf_matrix):
        """
        Element wise multiplication of tf and idf matrix to get the tf-idf matrix
        """
        tf_idf_matrix = {}

        for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

            tf_idf_table = {}

            for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                        f_table2.items()):  # here, keys are the same in both the table
                tf_idf_table[word1] = float(value1 * value2)

            tf_idf_matrix[sent1] = tf_idf_table

        return tf_idf_matrix

    def score_sentences(self , tf_idf_matrix) -> dict:
        """
        score a sentence by its word's TF
        Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
        :rtype: dict
        """

        sentenceValue = {}

        for sent, f_table in tf_idf_matrix.items():
            total_score_per_sentence = 0

            count_words_in_sentence = len(f_table)
            for word, score in f_table.items():
                total_score_per_sentence += score

            sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

        return sentenceValue
            