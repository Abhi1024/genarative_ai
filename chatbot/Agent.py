import os
import streamlit as st
from crewai import Agent, Task, Process, Crew
from langchain_google_genai import ChatGoogleGenerativeAI

def key():
    os.environ["GOOGLE_API_KEY"] = ""

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=key(), temperature=True, verbose=True)

st.title("Researcher Chatbot")

Researcher = Agent(
    role                = "Wikipedia and Various Search Engine Researcher.",
    goal                = "Find the meaningful and informative insights according to the {topic}.",
    verbose             = True,
    memory              = True,
    backstory           = ("Expert in finding the meaningful and sensible information of the {topic} and get the useful content."),
    tools               = [],
    llm                 = llm,
    allow_delegation    = True
)

Writer = Agent(
    role                = "Writer.",
    goal                = "Write the meaningful and sensible information insights according to the {topic}.",
    verbose             = True,
    memory              = True,
    backstory           = ("Get insight and write into the file."),
    tools               = [],
    llm                 = llm,
    allow_delegation    = True
)

Researcher_task = Task(
    description         = ("Identify the important point in the {topic} and final report should be clear and understandable."),
    expected_output     = "A comprensive and full 15 ponts paragraphs.",
    tools               = [],
    agent               = Researcher
)

Writer_task = Task(
    description         = ("Write the meaningful and informative insights of the {topic}. The article should be clear and proper understandable, enagging and positive points."),
    expected_output     = "A comprensive and full 12 points paragraphs.",
    tools               = [],
    agent              = Writer,
    async_execution     = False,
    output_file         = "Article.md",
)

crew = Crew(
    agents              = [Researcher,Writer],
    tasks               = [Researcher_task, Writer_task],
    process             = Process.sequential
)

Que = st.text_input("Please Enter the topic for research.")

if Que:
    result = crew.kickoff(inputs={"topic":Que})
    st.write(llm.invoke(result).content)
