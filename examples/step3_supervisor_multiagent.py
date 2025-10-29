"""
示例3：真正的Multi-Agent系统 - Supervisor模式
展示如何创建一个带有协调者的Multi-Agent系统

架构说明：
- Supervisor Agent：协调者，负责理解用户需求，决定调用哪个专业Agent
- Researcher Agent：研究专家，负责信息收集和分析
- Coder Agent：编程专家，负责编写和解释代码
- Writer Agent：写作专家，负责文档撰写和内容创作

这是真正的Multi-Agent系统，因为：
1. 每个Agent都是独立的，有自己的系统提示和能力
2. Supervisor会根据任务动态选择合适的Agent
3. 可以多轮调用不同的Agent直到任务完成
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

# 加载环境变量
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


# 定义路由选择的结构
class RouteResponse(BaseModel):
    """Supervisor的路由决策"""
    next_agent: Literal["researcher", "coder", "writer", "FINISH"]
    reason: str  # 选择该Agent的原因


# 定义状态
class AgentState(TypedDict):
    """Multi-Agent系统的状态"""
    messages: Annotated[list, lambda x, y: x + y]  # 消息历史
    next: str  # 下一个要执行的Agent


def create_supervisor_agent():
    """创建Supervisor Agent - 负责协调和路由"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    def supervisor(state: AgentState) -> AgentState:
        """Supervisor节点：决定下一步调用哪个Agent"""

        system_prompt = """你是一个任务协调者(Supervisor)。你需要分析用户的请求，并决定调用哪个专业Agent来处理。

可用的Agent：
- researcher: 研究专家，擅长信息收集、数据分析、知识整理
- coder: 编程专家，擅长编写代码、代码审查、技术问题解答
- writer: 写作专家，擅长文档撰写、内容创作、文字润色

你需要：
1. 分析用户的需求
2. 选择最合适的Agent
3. 如果任务已完成，选择FINISH
4. 说明你的选择理由

请以JSON格式回复，包含：
- next_agent: 选择的Agent名称（researcher/coder/writer/FINISH）
- reason: 选择的原因
"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        # 使用结构化输出
        structured_llm = llm.with_structured_output(RouteResponse)
        response = structured_llm.invoke(messages)

        print(f"\n[Supervisor决策] 选择: {response.next_agent}")
        print(f"[原因] {response.reason}\n")

        return {"next": response.next_agent}

    return supervisor


def create_researcher_agent():
    """创建Researcher Agent - 研究专家"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def researcher(state: AgentState) -> AgentState:
        """Researcher节点：处理研究类任务"""

        system_prompt = """你是一个专业的研究专家。你的职责是：
- 收集和分析相关信息
- 提供准确、全面的数据支持
- 整理和总结关键知识点
- 给出专业的研究结论

请基于用户的问题，提供详细的研究结果。"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)

        print(f"[Researcher Agent] {response.content[:100]}...\n")

        return {
            "messages": [response]
        }

    return researcher


def create_coder_agent():
    """创建Coder Agent - 编程专家"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    def coder(state: AgentState) -> AgentState:
        """Coder节点：处理编程类任务"""

        system_prompt = """你是一个专业的编程专家。你的职责是：
- 编写高质量、可维护的代码
- 解决技术问题和bug
- 提供代码优化建议
- 解释代码原理和最佳实践

请基于用户的需求，提供专业的编程解决方案。"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)

        print(f"[Coder Agent] {response.content[:100]}...\n")

        return {
            "messages": [response]
        }

    return coder


def create_writer_agent():
    """创建Writer Agent - 写作专家"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def writer(state: AgentState) -> AgentState:
        """Writer节点：处理写作类任务"""

        system_prompt = """你是一个专业的写作专家。你的职责是：
- 撰写清晰、优美的文章
- 进行文档整理和内容创作
- 提供文字润色和改进建议
- 确保内容结构合理、逻辑清晰

请基于用户的需求或之前Agent的输出，提供专业的写作内容。"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)

        print(f"[Writer Agent] {response.content[:100]}...\n")

        return {
            "messages": [response]
        }

    return writer


def route_after_agent(state: AgentState) -> str:
    """在Agent执行后决定路由"""
    # 每次Agent执行后，都返回supervisor重新评估
    return "supervisor"


def supervisor_route(state: AgentState) -> str:
    """Supervisor的路由决策"""
    next_agent = state.get("next", "FINISH")

    if next_agent == "FINISH":
        return END
    return next_agent


def create_multi_agent_system():
    """创建完整的Multi-Agent系统"""

    # 创建各个Agent
    supervisor = create_supervisor_agent()
    researcher = create_researcher_agent()
    coder = create_coder_agent()
    writer = create_writer_agent()

    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", researcher)
    workflow.add_node("coder", coder)
    workflow.add_node("writer", writer)

    # 添加边
    # START -> supervisor
    workflow.add_edge(START, "supervisor")

    # supervisor -> 根据决策路由到不同Agent或结束
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_route,
        {
            "researcher": "researcher",
            "coder": "coder",
            "writer": "writer",
            END: END
        }
    )

    # 各个Agent执行后都返回supervisor重新评估
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("coder", "supervisor")
    workflow.add_edge("writer", "supervisor")

    # 编译图
    graph = workflow.compile()

    return graph


def main():
    """主函数"""
    print("=" * 80)
    print("示例3：真正的Multi-Agent系统 - Supervisor模式")
    print("=" * 80)

    # 创建Multi-Agent系统
    agent_system = create_multi_agent_system()

    # 测试任务列表
    test_tasks = [
        "请帮我研究一下Python中的装饰器是什么，有什么用途",
        "写一个Python装饰器来计算函数执行时间",
        "基于上面的代码和研究，写一篇关于Python装饰器的教程文章"
    ]

    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'='*80}")
        print(f"任务 {i}: {task}")
        print(f"{'='*80}\n")

        # 调用Multi-Agent系统
        result = agent_system.invoke({
            "messages": [HumanMessage(content=task)],
            "next": ""
        })

        # 输出最终结果
        print(f"\n{'='*80}")
        print(f"任务 {i} 完成！")
        print(f"{'='*80}\n")

        # 显示完整对话历史
        if i == len(test_tasks):  # 最后一个任务显示完整历史
            print("\n完整对话历史：")
            for msg in result["messages"]:
                role = msg.__class__.__name__
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                print(f"\n[{role}] {content}")


if __name__ == "__main__":
    main()
