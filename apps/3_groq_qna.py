# LLM
# Tool -Google search tool
# Agent 
# Memory
# Streaming
# Web Interface
import os
import streamlit as st

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm=ChatGroq(model="openai/gpt-oss-20b",streaming=True)
search=GoogleSerperAPIWrapper()
tools=[search.run]

# memory=MemorySaver()
if "memory" not in st.session_state:    
    
    st.session_state.memory=MemorySaver()
    st.session_state.history=[{"role":"user","content":"query"}]

agent=create_agent(model=llm,tools=tools,checkpointer=st.session_state.memory,system_prompt="You are a helpful assistant that can answer questions using the Google Search API. Use the search tool to find information and provide accurate answers to the user's queries.")

# Building web interface
st.subheader("Groq QnA")
for message in st.session_state.history:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).write(content)
query=st.chat_input("Ask anything")

if query:
    st.chat_message("user").write(query)
    st.session_state.history.append({"role":"user","content":query})
    res = agent.stream(
        {"messages": [{"role": "user", "content":query}]},
        {"configurable": {"thread_id": "sankalp"}},
        stream_mode="messages"
    )
    
    ai_container=st.chat_message("assistant")
    
    with ai_container:
        space=st.empty()
        message=""
        
        for chunk in res:
            message+=chunk[0].content
            space.write(message)
        st.session_state.history.append({"role":"assistant","content":message})
      
        
    
    # answer=res["messages"][-1].content
    # st.chat_message("assistant").write(answer)
    # st.session_state.history.append({"role":"assistant","content":answer})
# print(res["messages"][-1].content)