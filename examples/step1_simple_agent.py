"""
示例1：最简单的单个Agent
展示如何创建一个基础的LangGraph Agent
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()


# 定义状态
class State(TypedDict):
    """Agent的状态，包含消息列表"""
    messages: Annotated[list, add_messages]


def create_simple_agent():
    """创建一个简单的Agent"""

    # 初始化LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    # 定义Agent节点
    def chatbot(state: State):
        """聊天机器人节点"""
        return {"messages": [llm.invoke(state["messages"])]}

    # 创建图
    graph_builder = StateGraph(State)

    # 添加节点
    graph_builder.add_node("chatbot", chatbot)

    # 添加边
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # 编译图
    graph = graph_builder.compile()

    return graph


def main():
    """主函数"""
    print("=" * 50)
    print("示例1：最简单的单个Agent")
    print("=" * 50)

    # 创建Agent
    agent = create_simple_agent()

    # 测试对话
    user_input = "你好！请介绍一下什么是Multi-Agent系统？"
    print(f"\n用户: {user_input}")

    # 调用Agent
    response = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    # 输出结果
    print(f"\nAgent: {response['messages'][-1].content}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
