import streamlit as st
from som_summarizer import summarizer
from nltk.tokenize import sent_tokenize

st.write('''
# Text Summarizer

#### Performs extractive summarization on text using SOM neural network
*****
''')

x = st.text_area('Input text to summarize', placeholder='Text goes here',height=300)

st.button('Analyze')

count = 0
for sent in sent_tokenize(x):
    count += 1

if count > 1 and x:
    n = st.slider('Select number of sentences in summary ',min_value=1,max_value=count)
    if st.button('Summarize'):
        summer = summarizer(250,n)
        with st.spinner('Generating summary...'):
            try:    
                summary = summer.generate_summary(x)
            except:
                st.error('Encountered an unexpected error, Please try again.')
            else:
                st.success('Summary Generated Successfully!')
                st.write(summary)
                st.download_button(label='Download Summary',data=summary,file_name='Your Summary.txt')
else:
    if x and count <= 1:
        st.error('Not enough sentences in text. Try again.')