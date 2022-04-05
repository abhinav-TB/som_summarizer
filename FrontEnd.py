import streamlit as st
from som_summarizer import summarizer
st.write('''
# SOMmarizer

#### Performs extractive summarization on text using SOM neural network
*****
''')

x = st.text_area('Input text to summarize', placeholder='Text goes here')

count = 0
for i in x:
    if i == '.':
        count+=1


if x :
    summer = summarizer(250)
    summary = summer.generate_summary(x)
    st.text(summary)
    st.download_button(label='Download Summary',data=summary,file_name='Your Summary.txt')