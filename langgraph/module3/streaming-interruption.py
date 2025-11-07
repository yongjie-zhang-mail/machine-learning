import os
import asyncio
from dotenv import load_dotenv

# .env file should contain your OPENROUTER_API_KEY
load_dotenv()
# https://smith.langchain.com
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langgraph"

from langchain_openai import ChatOpenAI
api_key = os.getenv("OPENROUTER_API_KEY")
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="tngtech/deepseek-r1t2-chimera:free",
    api_key=api_key,
    temperature=0    
)

from langgraph.graph import MessagesState
class State(MessagesState):
    summary: str

from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables.config import RunnableConfig
# Define the logic to call the model
def call_model(state: State, config: RunnableConfig):    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"
        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]    
    else:
        messages = state["messages"]
    
    # response = model.invoke(messages)
    response = model.invoke(messages, config)
    return {"messages": response}

def summarize_conversation(state: State):    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

from langgraph.graph import END
from typing_extensions import Literal
# Determine whether to end or summarize the conversation
def should_continue(state: State) -> Literal ["summarize_conversation", END]:
# def should_continue(state: State) -> str:
    """Return the next node to execute."""    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# 引入同目录下 util.py 中的工具类 Util，用于渲染图
from util import Util
Util.render_graph(g=graph, outfile="streaming-interruption.png", overwrite=True)

# state streaming example
# Create a thread
config = {"configurable": {"thread_id": "1"}}
# Start conversation; updates mode. 
for chunk in graph.stream({"messages": [HumanMessage(content="hi! I'm Lance")]}, config, stream_mode="updates"):
    # print(chunk)
    chunk['conversation']["messages"].pretty_print()

# Start conversation, again
config = {"configurable": {"thread_id": "2"}}
# Start conversation; values mode.
input_message = HumanMessage(content="hi! I'm Lance")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    for m in event['messages']:
        m.pretty_print()
    print("---"*25)

# token streaming example
async def _run_async_events():
    """Demonstrate async event streaming (moved into an async function to avoid SyntaxError)."""
    config = {"configurable": {"thread_id": "3"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")

    async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
        print(f"Node: {event['metadata'].get('langgraph_node','')}. Type: {event['event']}. Name: {event['name']}")

async def _run_async_token_streaming():
    node_to_stream = 'conversation'
    config = {"configurable": {"thread_id": "4"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")

    async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
        # Get chat model tokens from a particular node 
        if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
            print(event["data"])


async def _run_async_token_streaming2():
    node_to_stream = 'conversation'
    config = {"configurable": {"thread_id": "5"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")

    async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
        # Get chat model tokens from a particular node 
        if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
            data = event["data"]
            # print(data["chunk"].content, end="|")
            print(data["chunk"].content)




# if __name__ == "__main__":
#     # Run the async demo section
#     try:
#         # asyncio.run(_run_async_events())
#         # asyncio.run(_run_async_token_streaming())
#         # asyncio.run(_run_async_token_streaming2())
#     except KeyboardInterrupt:
#         print("Interrupted by user.")




