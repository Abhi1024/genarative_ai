import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def key():
    os.environ["GOOGLE_API_KEY"] = ""

st.title("Basic Chatbot Prototype.")
que = st.text_input("Please ask any question.")
# que = str(input("Please ask your question."))
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=key(), temperature=0.8)
# print(llm.invoke(que).content)
st.write(llm.invoke(que).content)
