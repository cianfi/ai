from langchain_core.prompts import ChatPromptTemplate
from local_llm import local_llm

llm = local_llm()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI Agent that likes to help others."
        ),
        (
            "human",
            "{input}"
        ),
    ]
)

chain = prompt | llm
chain_response = chain.invoke(
    {
        "input": "What is the current time and date?"
    }
)

print(chain_response)