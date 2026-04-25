from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
from langgraph.graph import StateGraph ,START,END
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated

class ChatState(BaseModel):
    message:Annotated[list,add_messages]

llm=ChatGroq(model="openai/gpt-oss-20b")

def chatBot(state:ChatState)-> ChatState:
    res=llm.invoke(state.message)
    state.message.append(res)
    return state


memory=InMemorySaver()

graph=StateGraph(ChatState)
graph.add_node("chatBot",chatBot)
graph.add_edge(START,"chatBot")
graph.add_edge("chatBot",END)

graph=graph.compile(checkpointer=memory)
config={"configurable":{"thread_id":"my-bot-1"}}

while True:
    query=input("User: ")
    if  query.lower()=="exit":
        print("Thanks for using me")
        exit()
    

    res=graph.invoke(
        {"message":[{"role":"user","content":query}]},
        config
        
    )
    ans=res["message"][-1].content
    print("Ai:",ans)