"""
示例2：多节点工作流
展示如何创建多个节点协作完成任务
场景：研究节点 + 写作节点的流水线工作流

注意：这不是真正的Multi-Agent系统，而是简单的多节点流水线。
真正的Multi-Agent系统请参考 step3_supervisor_multiagent.py
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 加载环境变量
load_dotenv()


# 定义状态
class State(TypedDict):
    """Multi-Agent系统的状态"""
    messages: Annotated[list, add_messages]
    research_result: str  # 研究结果
    article: str  # 最终文章


def create_multi_agent():
    """创建Multi-Agent系统"""

    # 初始化LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    # 研究助手Agent
    def researcher(state: State):
        """研究助手：负责收集和分析信息"""
        system_msg = SystemMessage(
            content="你是一个专业的研究助手。你的任务是针对用户的问题进行深入研究，"
                    "提供准确、全面的信息和数据支持。请用简洁的方式总结关键点。"
        )

        messages = [system_msg] + state["messages"]
        response = llm.invoke(messages)

        return {
            "research_result": response.content,
            "messages": [response]
        }

    # 写作助手Agent
    def writer(state: State):
        """写作助手：负责基于研究结果撰写文章"""
        system_msg = SystemMessage(
            content="你是一个专业的写作助手。你的任务是基于研究助手提供的信息，"
                    "撰写一篇结构清晰、逻辑严谨、易于理解的文章。"
        )

        # 构建写作提示
        writing_prompt = HumanMessage(
            content=f"基于以下研究结果撰写一篇文章：\n\n{state['research_result']}\n\n"
                    f"原始问题：{state['messages'][0].content}"
        )

        messages = [system_msg, writing_prompt]
        response = llm.invoke(messages)

        return {
            "article": response.content,
            "messages": [response]
        }

    # 创建图
    graph_builder = StateGraph(State)

    # 添加节点
    graph_builder.add_node("researcher", researcher)
    graph_builder.add_node("writer", writer)

    # 添加边：START -> 研究助手 -> 写作助手 -> END
    graph_builder.add_edge(START, "researcher")
    graph_builder.add_edge("researcher", "writer")
    graph_builder.add_edge("writer", END)

    # 编译图
    graph = graph_builder.compile()

    return graph


def main():
    """主函数"""
    print("=" * 50)
    print("示例2：多节点工作流")
    print("研究节点 + 写作节点")
    print("=" * 50)

    # 创建Multi-Agent系统
    agent = create_multi_agent()

    # 测试任务
    user_input = "请介绍一下Python在人工智能领域的应用"
    print(f"\n用户任务: {user_input}")
    print("\n" + "-" * 50)

    # 调用Agent系统
    response = agent.invoke({
        "messages": [HumanMessage(content=user_input)],
        "research_result": "",
        "article": ""
    })

    # 输出结果
    print("\n【研究助手的研究结果】")
    print(response['research_result'])
    print("\n" + "-" * 50)
    print("\n【写作助手的最终文章】")
    print(response['article'])
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
