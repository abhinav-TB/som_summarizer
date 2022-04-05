import streamlit as st
from som_summarizer import summarizer
import spacy

nlp = spacy.load('en_core_web_sm')

st.write('''
# SOMmarizer

#### Performs extractive summarization on text using SOM neural network
*****
''')

x = st.text_area('Input text to summarize', placeholder='Text goes here')

count = 0
doc = nlp(x)
for sent in doc.sents:
    count += 1

if x :
    n = st.slider('Number of sentences in summary :',min_value=1,max_value=count)
    if st.button('Make the Summary'):
        summer = summarizer(250,n)
        summary = summer.generate_summary(x)
        st.write(summary)
        st.download_button(label='Download Summary',data=summary,file_name='Your Summary.txt')