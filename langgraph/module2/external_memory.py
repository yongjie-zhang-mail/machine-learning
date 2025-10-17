import os
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
# Define the logic to call the model
def call_model(state: State):    
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
    
    response = model.invoke(messages)
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
# def should_continue(state: State) -> Literal ["summarize_conversation", END]:
def should_continue(state: State) -> str:
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
import sqlite3
# In memory
# conn = sqlite3.connect(":memory:", check_same_thread = False)
# pull file if it doesn't exist and connect to local db
# mkdir -p state_db && [ ! -f state_db/example.db ] && wget -P state_db https://github.com/langchain-ai/langchain-academy/raw/main/module-2/state_db/example.db
db_path = "state_db/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
# Here is our checkpointer 
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver(conn)

graph = workflow.compile(checkpointer=memory)

# 引入同目录下 util.py 中的工具类 Util，用于渲染图
from util import Util
Util.render_graph(g=graph, outfile="external_memory.png", overwrite=True)

# Create a thread
config = {"configurable": {"thread_id": "1"}}

# Start conversation
input_message = HumanMessage(content="hi! I'm Lance")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="what's my name?")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="i like the 49ers!")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

# 查看当前图的状态
config = {"configurable": {"thread_id": "1"}}
graph_state = graph.get_state(config)
print("Current graph state: \n", graph_state)



