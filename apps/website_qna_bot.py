from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from time import sleep

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.vectorstores import InMemoryVectorStore


# Load API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Website QnA Bot")

st.subheader("🌐 Customer Support QnA Bot")

# Session state initialization
if "web_loaded" not in st.session_state:
    st.session_state.web_loaded = False

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------
# Process URLs
# ----------------------------
def process_urls(urls):

    alldocs = []

    for url in urls:

        url = url.strip()

        # Validate URL
        if not url.startswith(("http://", "https://")):
            st.warning(f"Skipped invalid URL: {url}")
            continue

        try:
            loader = WebBaseLoader(web_path=url)

            docs = loader.load()

            alldocs.extend(docs)

            st.success(f"Loaded: {url}")

        except Exception as e:
            st.error(f"Error loading {url}")
            st.error(str(e))

    # Check if any data loaded
    if not alldocs:
        st.error("No website content found")
        return

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splitted_data = splitter.split_documents(alldocs)

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=GOOGLE_API_KEY
    )

    # Vector store
    vector_db = InMemoryVectorStore.from_documents(
        splitted_data,
        embeddings
    )

    st.session_state.vector_db = vector_db
    st.session_state.web_loaded = True


# ----------------------------
# Website loading section
# ----------------------------

if not st.session_state.web_loaded:

    urls = st.text_area(
        "Enter URLs (one URL per line)"
    )

    if st.button("Load Websites"):

        url_list = urls.strip().split("\n")

        with st.spinner(
            "Loading websites and processing..."
        ):

            process_urls(url_list)

            if st.session_state.web_loaded:

                st.success(
                    "Websites loaded successfully"
                )

                sleep(2)

                st.rerun()


# ----------------------------
# Chat section
# ----------------------------

if (
    st.session_state.web_loaded
    and st.session_state.vector_db
):

    # Display chat history
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    query = st.chat_input(
        "Ask a question about the website content"
    )

    if query:

        # Display user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        # Similarity search
        records = (
            st.session_state.vector_db
            .similarity_search(
                query,
                k=6
            )
        )

        context = ""

        for chunk in records:
            context += (
                chunk.page_content
                + "\n\n"
            )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY
        )

        prompt = f"""
You are a website support assistant.

Answer only using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

        with st.spinner("Thinking..."):

            response = llm.invoke(
                prompt
            )

        with st.chat_message(
            "assistant"
        ):
            st.markdown(
                response.content
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.content
            }
        )