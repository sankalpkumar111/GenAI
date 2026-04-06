from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.agents import create_agent
from langchain.tools import tool   # lowercase decorator, works in your version

load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Google Search wrapper
search = GoogleSearchAPIWrapper()

# Wrap search as a tool using decorator
@tool
def google_search(query: str) -> str:
    """Search Google for up-to-date information"""
    return search.run(query)

# Create agent
agent = create_agent(
    model=llm,  
    tools=[google_search],  # pass the decorated function
    system_prompt="You are a helpful assistant. Use Google search when needed."
)

# Chat loop
while True:
    query = input("Ask me anything (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    response = agent.invoke({"input": query, "content": query})

    print("Answer:", response["output"])
