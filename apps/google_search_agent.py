from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper

search=GoogleSerperAPIWrapper()
llm=ChatGroq(model="openai/gpt-oss-20b")

tools=[search.run]

system_prompt="You are a helpful assistant that can answer questions using the Google Search API. Use the search tool to find information and provide accurate answers to the user's queries."

agent=create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
    )

while True:
    if query.lower()=="exit" or query.lower()=="quit" or query.lower()=="bye":
        print("Existing")
        break
    query=input("You: ")
    res=agent.invoke({
        "messages":[
            {"role":"user","content":query}
            ]
        })

    answer=res["messages"][-1].content
    print("AI: ",answer)

# res=llm.invoke(query)
# print(res.content)
