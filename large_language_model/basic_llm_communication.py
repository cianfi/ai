from langchain_core.messages import AIMessage

from local_llm import local_llm

llm = local_llm()

messages = [
    (
        "system",
        "You are an AI Agent that likes to help others learn about geography."
    ),
    (
        "human",
        "What is the population of India and China"
    ),
]

ai_msg = llm.invoke(messages)
print(ai_msg)