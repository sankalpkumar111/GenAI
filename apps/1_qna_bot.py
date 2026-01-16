from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
st.title("🤖 Q&A Bot")
st.markdown("Ask any question and get answers powered by Google Gemini")

if "messages" not in st.session_state:
    st.session_state.messages=[]
    
for message in st.session_state.messages:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).write(content)
    
query=st.chat_input("Ask me anything...")
if query:
    st.session_state.messages.append({"role":"user", "content":query})
    st.chat_message("user").write(query)
    
    response=llm.invoke(query)
    st.chat_message("assistant").write(response.content)
    st.session_state.messages.append({"role":"assistant", "content":response.content})




# while True:
#     que=input("You:")
#     if que.lower()=="exit" or que.lower()=="quit" or que.lower()=="bye":
#         print("Exiting...")
#         break
#     res=llm.invoke(que)
#     print(res.content,"\n")