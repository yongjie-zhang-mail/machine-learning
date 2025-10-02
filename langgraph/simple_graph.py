from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str


def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" sad!"}


import random
from typing import Literal

def decide_mood(state) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"


from langgraph.graph import StateGraph, START, END
import os

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()


def render_graph(g, outfile: str = "simple_graph.png", overwrite: bool = False) -> None:
    """生成 Mermaid PNG；若文件已存在且未指定 overwrite，则跳过。"""
    if not overwrite and os.path.isfile(outfile):
        print(f"Skip existing: {outfile} (set overwrite=True to regenerate)")
        return
    with open(outfile, "wb") as f:
        f.write(g.get_graph().draw_mermaid_png())
    print(f"Saved graph PNG: {outfile}")


# Render the graph to a PNG file (will skip if exists)
render_graph(graph)

# Invoke the graph with initial state
resp = graph.invoke({"graph_state" : "Hi, this is Lance."})
print("Final response:", resp)



