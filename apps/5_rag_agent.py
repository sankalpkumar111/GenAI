from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st



###  Data in st session 
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded=False

if "agent" not in  st.session_state:
    st.session_state.agent=None

if "vector_store" not in st.session_state:
    st.session_state.vector_store=None

if "message" not in st.session_state:
    st.session_state.message=[]




def process_document(path):

    #### Document load
    loader = PyPDFDirectoryLoader(path)

    docs = loader.load()

    ###  Split into multiple chunks
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs = splitter.split_documents(docs)

    # Embeddings and Vector stores
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_db = InMemoryVectorStore.from_documents(documents=docs,embedding=embeddings)

    # Create agent -tool,llm,prompt

    llm=ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def retrieve_context(query:str):
        """
        Retrieve document relevant to a query from  the knowledge base.
        """
        context=""
        docs=vector_db.similarity_search(query,k=3)
        for doc in docs:
            context += doc.page_content + "\n\n"
        return context

    system_prompt = """
    You are a helpful assistant that can answer questions 
    My knowledge base consists of the details  from the upload documents
    Always use the 'retrieve_context' tools for question requiring  external knowledge
    """


    memory=InMemorySaver()

    agent=create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
        )
    st.session_state.agent=agent
    st.session_state.document_uploaded=True
    


# Upload ui

if not st.session_state.document_uploaded:
    uploaded=st.file_uploader(label="Select Pdf File",type=["pdf"],accept_multiple_files=True)
    if uploaded:
        with st.spinner("Uploading..."):
            path="./doc_files/"
            for file in uploaded:
                with open(path+file.name,"wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()
        
            

# Chat ui
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.message:
        role=message.get("role")
        content=message.get("content")
        st.chat_message(role).write(content)
    query=st.chat_input("Ask me anything related to the uploaded document")
    
    
    if query:
        st.session_state.message.append({"role":"user","content":query})
        st.chat_message("user").write(query)

        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable":{"thread_id":"sankalp"}}
        )

        answer = response["messages"][-1].content

        st.chat_message("assistant").write(answer)
        st.session_state.message.append({"role":"assistant","content":answer})
            
        

# Cmd based
# while True:
#     query=input("User: ")
#     if query.lower() =="quit":
#         break
#     res = agent.invoke({"messages":[{"role":"user","content":query}]}, 
#                  {"configurable": {"thread_id": "sankalp"}})
#     result=res["messages"][-1].content
#     print("AI: ",result)