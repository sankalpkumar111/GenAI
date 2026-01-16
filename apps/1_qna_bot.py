# Load environment variables from .env file
# This is where API keys are stored (never hardcode keys)
from dotenv import load_dotenv
load_dotenv()

# Import Gemini LLM wrapper from LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Streamlit is used to create the chat UI
import streamlit as st


# Create Gemini LLM object
# gemini-2.5-flash is fast and good for chat applications
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# App title and description shown on the UI
st.title("🤖 Q&A Bot")
st.markdown("Ask any question and get answers powered by Google Gemini")


# session_state is used to store chat history
# Streamlit reruns the script every time, so without this chat will reset
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display old messages when app reloads
# This keeps the conversation visible
for message in st.session_state.messages:
    role = message["role"]      # user or assistant
    content = message["content"]
    st.chat_message(role).write(content)


# Chat input box for user
query = st.chat_input("Ask me anything...")


# If user enters a question
if query:
    # Store user message in session
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Show user message on UI
    st.chat_message("user").write(query)

    # Send question to Gemini model
    response = llm.invoke(query)

    # Show AI response on UI
    st.chat_message("assistant").write(response.content)

    # Store AI response in session
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.content
    })


# Below code is for terminal-based chat (not Streamlit)
# Useful only for testing, so kept commented

# while True:
#     que = input("You: ")
#     if que.lower() in ["exit", "quit", "bye"]:
#         print("Exiting...")
#         break
#     res = llm.invoke(que)
#     print(res.content, "\n")
