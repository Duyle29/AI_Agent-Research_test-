import os
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain.agents import AgentExecutor

from tools import search_tool, wiki_tool, save_tool


# load environment variables
load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# Gemini model

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# Output parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Bạn là một trợ lý nghiên cứu viết tóm tắt nghiên cứu bằng tiếng Việt.

Sử dụng công cụ nếu cần.

Trả về CHỈ JSON.

{format_instructions}
"""
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())


# Tools
tools = [search_tool, wiki_tool, save_tool]


# Create agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


# Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


# User input
query = input("Bạn muốn tìm hiểu cái gì? ")


# Run agent
raw_response = agent_executor.invoke(
    {"input": query}
)


# Parse output
try:
    structured_response = parser.parse(raw_response["output"])
    print("\nStructured Response:")
    print(structured_response)

except Exception as e:
    print("\nParsing Error:", e)
    print("\nRaw Response:")
    print(raw_response["output"])
