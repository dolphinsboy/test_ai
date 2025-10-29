"""
示例4：带工具调用的Multi-Agent系统
展示Agent如何使用工具来增强能力

这个示例包含：
- Calculator Agent：可以使用计算器工具的数学专家
- Search Agent：可以使用搜索工具的信息检索专家（模拟）
- Supervisor：协调者，决定使用哪个Agent

工具调用是Multi-Agent系统的重要特性：
- Agent不仅可以生成文本，还可以执行实际操作
- 通过工具扩展Agent的能力边界
- 实现真正的任务自动化
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel

# 加载环境变量
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")



# ========== 定义工具 ==========

@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式的结果

    Args:
        expression: 数学表达式，如 "2 + 2" 或 "3 * 4 + 5"

    Returns:
        计算结果
    """
    try:
        # 注意：在生产环境中应该使用更安全的表达式求值方式
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def search_web(query: str) -> str:
    """
    搜索网络信息（模拟）

    Args:
        query: 搜索查询词

    Returns:
        搜索结果
    """
    # 这里是模拟搜索，实际应该调用真实的搜索API
    mock_results = {
        "python": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。它以简洁、易读的语法著称。",
        "langgraph": "LangGraph是LangChain推出的框架，用于构建有状态的多智能体应用。它使用图结构来定义Agent工作流。",
        "ai": "人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
    }

    # 简单的关键词匹配
    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return f"搜索结果：{value}"

    return f"未找到关于'{query}'的相关信息（这是模拟搜索）"


# ========== 定义状态 ==========

class RouteResponse(BaseModel):
    """Supervisor的路由决策"""
    next_agent: Literal["calculator_agent", "search_agent", "FINISH"]
    reason: str


class AgentState(TypedDict):
    """系统状态"""
    messages: Annotated[list, lambda x, y: x + y]
    next: str


# ========== 创建Agents ==========

def create_calculator_agent():
    """创建计算器Agent - 可以使用计算工具"""

    # 创建带工具的LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    tools = [calculator]
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: AgentState):
        """Calculator Agent节点"""
        system_prompt = """你是一个数学计算专家。当用户需要进行数学计算时：
1. 使用calculator工具来计算
2. 解释计算过程
3. 给出最终答案

你有calculator工具可以使用。"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)

        print(f"[Calculator Agent] 调用工具: {response.tool_calls if response.tool_calls else '无'}")

        return {"messages": [response]}

    return agent, tools


def create_search_agent():
    """创建搜索Agent - 可以使用搜索工具"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    tools = [search_web]
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: AgentState):
        """Search Agent节点"""
        system_prompt = """你是一个信息检索专家。当用户需要查找信息时：
1. 使用search_web工具搜索相关信息
2. 总结和整理搜索结果
3. 给出准确的答案

你有search_web工具可以使用。"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)

        print(f"[Search Agent] 调用工具: {response.tool_calls if response.tool_calls else '无'}")

        return {"messages": [response]}

    return agent, tools


def create_supervisor():
    """创建Supervisor"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    def supervisor(state: AgentState):
        """Supervisor节点"""
        system_prompt = """你是任务协调者。分析用户请求，选择合适的Agent：

- calculator_agent: 处理数学计算、算术问题
- search_agent: 处理信息查询、知识问题
- FINISH: 任务已完成

返回JSON格式：{next_agent: "...", reason: "..."}"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        structured_llm = llm.with_structured_output(RouteResponse)
        response = structured_llm.invoke(messages)

        print(f"\n[Supervisor] 选择: {response.next_agent} - {response.reason}\n")

        return {"next": response.next_agent}

    return supervisor


def should_continue(state: AgentState) -> str:
    """判断是否继续"""
    messages = state["messages"]
    last_message = messages[-1]

    # 如果最后一条消息包含工具调用，执行工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # 否则返回supervisor
    return "supervisor"


def supervisor_route(state: AgentState) -> str:
    """Supervisor路由"""
    next_agent = state.get("next", "FINISH")
    if next_agent == "FINISH":
        return END
    return next_agent


def create_multi_agent_system():
    """创建完整的Multi-Agent系统"""

    # 创建Agents
    supervisor = create_supervisor()
    calculator_agent, calc_tools = create_calculator_agent()
    search_agent, search_tools = create_search_agent()

    # 合并所有工具
    all_tools = calc_tools + search_tools
    tool_node = ToolNode(all_tools)

    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("calculator_agent", calculator_agent)
    workflow.add_node("search_agent", search_agent)
    workflow.add_node("tools", tool_node)

    # 添加边
    workflow.add_edge(START, "supervisor")

    # Supervisor路由
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_route,
        {
            "calculator_agent": "calculator_agent",
            "search_agent": "search_agent",
            END: END
        }
    )

    # Agent执行后的路由
    workflow.add_conditional_edges(
        "calculator_agent",
        should_continue,
        {
            "tools": "tools",
            "supervisor": "supervisor"
        }
    )

    workflow.add_conditional_edges(
        "search_agent",
        should_continue,
        {
            "tools": "tools",
            "supervisor": "supervisor"
        }
    )

    # 工具执行后返回相应的Agent
    def route_after_tools(state: AgentState) -> str:
        """工具执行后路由回调用它的Agent"""
        messages = state["messages"]
        # 查找最近的AI消息来确定是哪个Agent
        for msg in reversed(messages[:-1]):
            if isinstance(msg, AIMessage):
                # 这里简化处理，实际应该更智能
                return "calculator_agent"  # 或根据上下文决定
        return "supervisor"

    # 简化：工具执行后返回supervisor重新评估
    workflow.add_edge("tools", "supervisor")

    # 编译
    graph = workflow.compile()
    
    # 保存图到本地文件
    try:
        # 保存为PNG图片
        png_data = graph.get_graph().draw_mermaid_png()
        with open("agent_workflow.png", "wb") as f:
            f.write(png_data)
        print("✅ 工作流图已保存为 agent_workflow.png")
        
    except Exception as e:
        print(f"⚠️ 保存图时出错: {e}")
        print("图生成功能可能需要额外的依赖包")

    return graph


def main():
    """主函数"""
    print("=" * 80)
    print("示例4：带工具调用的Multi-Agent系统")
    print("=" * 80)

    # 创建系统
    agent_system = create_multi_agent_system()

    # 测试任务
    test_tasks = [
        "计算 (123 + 456) * 789 的结果",
        "搜索一下什么是LangGraph",
        "计算 100 的平方根是多少（提示：100 ** 0.5）"
    ]

    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'='*80}")
        print(f"任务 {i}: {task}")
        print(f"{'='*80}\n")

        result = agent_system.invoke({
            "messages": [HumanMessage(content=task)],
            "next": ""
        })

        # 输出最终结果
        final_message = result["messages"][-1]
        print(f"\n最终回复: {final_message.content}\n")


if __name__ == "__main__":
    main()
