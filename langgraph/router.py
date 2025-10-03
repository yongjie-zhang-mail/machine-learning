from langchain_openai import ChatOpenAI
import os

api_key = os.getenv("OPENROUTER_API_KEY")  # 确保已 export
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="x-ai/grok-4-fast:free",
    api_key=api_key,
    temperature=0    
)

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()

# View
from util import Util
Util.render_graph(g=graph, outfile="router.png", overwrite=True)

# Test graph
from langchain_core.messages import HumanMessage
messages = [HumanMessage(content="Hello, what is 3 multiplied by 5?")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()

messages = [HumanMessage(content="Hello")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()


