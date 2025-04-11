import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun # DuckDuckGoSearchRun -> helps to search anything on the internet
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler ## allows to communicate with all these tools within themselves
import os
from dotenv import load_dotenv

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")


st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {
            "role":"assistant",
            "content":"Hi,I'm a chatbot who can search the web. How can I help you?"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    # := is a walrus operator, meaning if user typed something save it to prompt and proceed
    st.session_state.messages.append({"role":"user","content":prompt}) # Stores the message in session so that conversation persists.
    st.chat_message("user").write(prompt) # Displays the user message in the chat area.

    llm=ChatGroq(
        groq_api_key=api_key,
        model_name="Llama3-8b-8192",
        streaming=True # streaming=True means the assistant will respond gradually, not all at once.
    )

    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # it decides when to use tools and does step-by-step reasoning before answering.
        #AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION -> Does a reasoning on the basis of chat history and chat context
        handling_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False
        )

        response=search_agent.run(
            st.session_state.messages,
            callbacks=[st_cb]
            # st_cb is a callback handler to display the thought process or tool usage live in Streamlit.
        )
        # Callback lets you see live steps if tools are being used.

        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)

        # Saves the assistant’s final answer in the chat history.
        # Displays the final response below the message block

