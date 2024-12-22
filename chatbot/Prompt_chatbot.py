import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.sequential import SequentialChain
from langchain.chains.llm import LLMChain

def key():
    os.environ["GOOGLE_API_KEY"] = ""

st.title("Demo Chatbot")
que = st.text_input("Ask Any Thing ... ")
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=key(), temperature=0.8, verbose= True,)

first_input = PromptTemplate(
    input_variables=["name"],
    template="Name of this person is {name}."
)

chain1 = LLMChain(llm=llm, prompt=first_input, verbose=True, output_key="Nicknames")

second_input = PromptTemplate(
    input_variables=["Nicknames"],
    template="This person was born in {Nicknames}."
)

chain2 = LLMChain(llm=llm, prompt=second_input, verbose= True, output_key="Place")

third_input = PromptTemplate(
    input_variables=["Place"],
    template="The nickname of the {Place}."
)

chain3 = LLMChain(llm=llm, prompt=third_input, verbose=True)

sqchain = SequentialChain(chains=[chain1,chain2,chain3], verbose=True, input_variables=["name"], output_variables=["Nicknames","Place"])

# st.write(sqchain.run(que))

st.write(sqchain.invoke(que))

# st.write(llm.invoke(que).content)

# if que:
    # st.write(llm.invoke(search).content)
    # st.write(chain.run(search))
    # st.write(parentchain.run(search))
    # st.write(sqchain({"name":que}))

# st.write(llm.invoke(que).content)
