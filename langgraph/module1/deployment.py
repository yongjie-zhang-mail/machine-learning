import asyncio
from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage

# URL of the local development server (adjust if your server runs elsewhere)
URL = "http://127.0.0.1:2024"
client = get_client(url=URL)


async def main():
    # Search all hosted graphs
    assistants = await client.assistants.search()
    # (Optional) you can inspect assistants if needed
    print(assistants)

    # We create a thread for tracking the state of our run
    thread = await client.threads.create()

    from langchain_core.messages import HumanMessage  # kept (minimal change)

    # Input
    input = {"messages": [HumanMessage(content="Multiply 3 by 2.")]}

    # Stream
    async for chunk in client.runs.stream(
            thread['thread_id'],
            "agent_memory",
            input=input,
            stream_mode="values",
        ):
        if chunk.data and chunk.event != "metadata":
            print(chunk.data['messages'][-1])


if __name__ == "__main__":
    asyncio.run(main())