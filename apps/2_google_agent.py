from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver  


load_dotenv()

llm=ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
memory=MemorySaver()

agent=create_agent(
    model=llm,
    tools=[search.run],
    checkpointer=memory,
    system_prompt="You are a helpful assistant that can answer questions using the Google Search API. Use the search tool to find information and provide accurate answers to the user's queries."
    
)

while True:
    query=input("You: ")
    if query.lower()=="exit":
        print("Existing")
        break
    res = agent.invoke({"messages":[{"role":"user","content":query}]}, 
                 {"configurable": {"thread_id": "sankalp"}})
    print("AI ",res["messages"][-1].content)