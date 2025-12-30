import asyncio
import json

from langgraph_sdk import get_client
from langgraph.pregel.remote import RemoteGraph
from langchain_core.messages import convert_to_messages
from langchain_core.messages import HumanMessage, SystemMessage
from uuid import uuid4


async def seed_initial_todos(client, thread, graph_name, config):
    """Seed baseline ToDos for the configured user."""

    # config = {"configurable": {"user_id": "Test"}}
    # 获取 user_id 的值
    user_id = config["configurable"]["user_id"]

    # 检查存储中是否已经存在 ToDo 项目, 如果不存在则添加初始 ToDo 项目
    items = await client.store.search_items(
        ("todo", "general", user_id),
        limit=5,
        offset=0
    )

    print("\n\n" + "-" * 40 + "  打印目前 store 中的 items  " + "-" * 40 + "\n\n")
    # 将 items 对象使用 json 格式打印出来
    print(json.dumps(items, ensure_ascii=False, indent=2))

    # items['items'] is a list of items in the store
    # 判断 items['items'] 数组包含的元素个数是否大于0
    if len(items['items']) == 0:
        user_input = (
            "Add a ToDo to finish booking travel to Hong Kong by end of next week. "
            "Also, add a ToDo to call parents back about Thanksgiving plans."
        )
        run = await client.runs.create(
            thread["thread_id"],
            graph_name,
            input={"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        await client.runs.join(thread["thread_id"], run["run_id"])


async def stream_chat(client, graph_name, config):
    """Create a fresh thread, summarize ToDos, stream priorities, and return the thread."""

    thread = await client.threads.create()
    user_input = "Give me a summary of all ToDos."
    run = await client.runs.create(
        thread["thread_id"],
        graph_name,
        input={"messages": [HumanMessage(content=user_input)]},
        config=config,
    )

    await client.runs.join(thread["thread_id"], run["run_id"])

    print("\n\n" + "-" * 40 + "  流式输出  " + "-" * 40 + "\n\n")

    user_input = "What ToDo should I focus on first."
    async for chunk in client.runs.stream(
        thread["thread_id"],
        graph_name,
        input={"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages-tuple",
    ):
        if chunk.event == "messages":
            print(
                "".join(
                    data_item["content"] for data_item in chunk.data if "content" in data_item
                ),
                end="",
                flush=True,
            )

    # 打印出两行换行 和 一行 "-" 分隔线
    print("\n\n" + "-" * 40 + "  打印 tread_state 中所有的 messages  " + "-" * 40 + "\n\n")
    
    thread_state = await client.threads.get_state(thread["thread_id"])
    for message in convert_to_messages(thread_state["values"]["messages"]):
        message.pretty_print()

    return thread


async def human_in_the_loop(client, thread, graph_name, config):
    """Fork a recent checkpoint from thread history and replay it."""

    states = await client.threads.get_history(thread["thread_id"])
    to_fork = states[-2]

    forked_input = {
        "messages": HumanMessage(
            content="Give me a summary of all ToDos that need to be done in the next week.",
            id=to_fork["values"]["messages"][0]["id"],
        )
    }

    forked_config = await client.threads.update_state(
        thread["thread_id"],
        forked_input,
        checkpoint_id=to_fork["checkpoint_id"],
    )

    print("\n\n" + "-" * 40 + "  human_in_the_loop 更改某个 checkpoint  " + "-" * 40 + "\n\n")

    async for chunk in client.runs.stream(
        thread["thread_id"],
        graph_name,
        input=None,
        config=config,
        checkpoint_id=forked_config["checkpoint_id"],
        stream_mode="messages-tuple",
    ):
        if chunk.event == "messages":
            print(
                "".join(
                    data_item["content"] for data_item in chunk.data if "content" in data_item
                ),
                end="",
                flush=True,
            )


async def manage_store_items(client):
    """Demonstrate basic put/search/delete operations against the store."""

    await client.store.put_item(
        ("testing", "Test"),
        key=str(uuid4()),
        value={"todo": "Test SDK put_item"},
    )

    items = await client.store.search_items(("testing", "Test"), limit=5, offset=0)

    print("\n\n" + "-" * 40 + "  打印增加到 store 中的 items  " + "-" * 40 + "\n\n")

    print(json.dumps(items, ensure_ascii=False, indent=2))

    for item in items["items"]:
        await client.store.delete_item(("testing", "Test"), key=item["key"])

    items = await client.store.search_items(("testing", "Test"), limit=5, offset=0)

    print("\n\n" + "-" * 40 + "  打印从 store 中删除后的 items  " + "-" * 40 + "\n\n")

    print(json.dumps(items, ensure_ascii=False, indent=2))


async def main() -> None:
    """Create a thread, invoke the deployed graph, and wait for completion."""

    # Connect via SDK
    url_for_cli_deployment = "http://localhost:8123"
    client = get_client(url=url_for_cli_deployment)

    graph_name = "memory_agent"
    config = {"configurable": {"user_id": "Test"}}
    
    thread = await client.threads.create()

    # Seed initial ToDos
    await seed_initial_todos(client, thread, graph_name, config)

    # Stream chat interaction
    thread = await stream_chat(client, graph_name, config)

    # Human-in-the-loop fork and replay
    await human_in_the_loop(client, thread, graph_name, config)

    # Manage store items
    await manage_store_items(client)


if __name__ == "__main__":
    asyncio.run(main())


