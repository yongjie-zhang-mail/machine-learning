import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"

from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage
# messages = [AIMessage(f"So you said you were researching ocean mammals?", name="Bot")]
# messages.append(HumanMessage(f"Yes, I know about whales. But what others should I learn about?", name="Lance"))

# for m in messages:
#     m.pretty_print()


from langchain_openai import ChatOpenAI
api_key = os.getenv("OPENROUTER_API_KEY")  # 确保已 export
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat-v3.1:free",
    api_key=api_key,
    temperature=0
)

# resp = llm.invoke(messages)
# pprint(resp)


from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import RemoveMessage

# Reducer Solution: 
# # Nodes
# def filter_messages(state: MessagesState):
#     # Delete all but the 2 most recent messages
#     delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
#     return {"messages": delete_messages}

# def chat_model_node(state: MessagesState):    
#     return {"messages": [llm.invoke(state["messages"])]}

# # Build graph
# builder = StateGraph(MessagesState)
# builder.add_node("filter", filter_messages)
# builder.add_node("chat_model", chat_model_node)
# builder.add_edge(START, "filter")
# builder.add_edge("filter", "chat_model")
# builder.add_edge("chat_model", END)
# graph = builder.compile()


# Filter Messages Solution: 
# # Node
# def chat_model_node(state: MessagesState):
#     return {"messages": [llm.invoke(state["messages"][-1:])]}

# # Build graph
# builder = StateGraph(MessagesState)
# builder.add_node("chat_model", chat_model_node)
# builder.add_edge(START, "chat_model")
# builder.add_edge("chat_model", END)
# graph = builder.compile()


# Trim Messages Solution:
from langchain_core.messages import trim_messages

def rough_token_counter(msgs):
    text = "\n".join(str(m.content) for m in msgs)
    return max(1, len(text) // 4)   # 粗略：4 chars ≈ 1 token


# Node
def chat_model_node(state: MessagesState):
    messages = trim_messages(
            state["messages"],
            max_tokens=10,
            strategy="last",
            token_counter=rough_token_counter,
            allow_partial=False,
        )
    return {"messages": [llm.invoke(messages)]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()



from util import Util
Util.render_graph(graph, "trim_filter_messages.png", overwrite=True)


# Message list with a preamble
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

messages.append(HumanMessage(f"Tell me more about Narwhals!", name="Lance"))

messages.append(HumanMessage(f"Tell me where Orcas live!", name="Lance"))

# Example of trimming messages
print(trim_messages(
    messages,
    max_tokens=10,
    strategy="last",
    token_counter=rough_token_counter,
    allow_partial=False
))



# Invoke
# output = graph.invoke({'messages': messages})
# for m in output['messages']:
#     m.pretty_print()

# Invoke, using message trimming in the chat_model_node 
messages_out_trim = graph.invoke({'messages': messages})
for m in messages_out_trim['messages']:
    m.pretty_print()
