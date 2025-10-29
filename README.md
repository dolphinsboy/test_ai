# Multi-Agent Demo

基于LangGraph构建的Multi-Agent系统演示项目，面向LLM小白的教学示例。

## 项目简介

本项目通过Step by Step的方式，展示如何使用LangGraph构建多智能体（Multi-Agent）系统。每个示例都包含详细的注释和说明，帮助初学者理解Multi-Agent架构的核心概念。

## 技术栈

- **Python**: 3.12
- **包管理**: uv
- **核心框架**: LangGraph 1.0.1
- **LLM**: OpenAI GPT-4

## 快速开始

### 1. 环境准备

确保已安装Python 3.12和uv包管理器。

### 2. 克隆项目

```bash
git clone <your-repo-url>
cd test_ai
```

### 3. 安装依赖

```bash
uv sync
```

### 4. 配置环境变量

复制`.env.example`文件为`.env`，并填入你的OpenAI API Key：

```bash
cp .env.example .env
```

编辑`.env`文件：

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. 运行示例

#### 示例1：简单的单个Agent

```bash
uv run examples/step1_simple_agent.py
```

这个示例展示了如何创建一个最基础的LangGraph Agent：
- State（状态）的定义
- Node（节点）的创建
- Graph（图）的构建

#### 示例2：多节点工作流

```bash
uv run examples/step2_multi_agent.py
```

这个示例展示了多个节点的流水线协作：
- 研究节点：负责信息收集和分析
- 写作节点：负责基于研究结果撰写文章
- 节点之间的数据传递

**注意**：这不是真正的Multi-Agent系统，只是简单的节点流水线。

#### 示例3：真正的Multi-Agent系统（Supervisor模式）⭐

```bash
uv run examples/step3_supervisor_multiagent.py
```

这是真正的Multi-Agent系统示例：
- **Supervisor Agent**：协调者，根据任务动态选择合适的Agent
- **Researcher Agent**：研究专家，独立的智能体
- **Coder Agent**：编程专家，独立的智能体
- **Writer Agent**：写作专家，独立的智能体
- **动态路由**：根据任务需求智能选择Agent
- **多轮协作**：可以多次调用不同Agent完成复杂任务

#### 示例4：带工具调用的Multi-Agent系统⭐

```bash
uv run examples/step4_agent_with_tools.py
```

展示Agent如何使用工具增强能力：
- **Calculator Agent**：使用计算器工具的数学专家
- **Search Agent**：使用搜索工具的信息检索专家
- **工具调用**：Agent不仅能对话，还能执行实际操作
- **任务自动化**：通过工具扩展Agent的能力边界

## 项目结构

```
test_ai/
├── src/                               # 源代码目录
│   ├── agents/                        # Agent定义
│   ├── tools/                         # 工具函数
│   └── utils/                         # 工具类
├── examples/                          # 示例代码
│   ├── step1_simple_agent.py          # 示例1：单个Agent
│   ├── step2_multi_agent.py           # 示例2：多节点流水线
│   ├── step3_supervisor_multiagent.py # 示例3：真正的Multi-Agent⭐
│   └── step4_agent_with_tools.py      # 示例4：带工具调用的Agent⭐
├── .env.example                       # 环境变量示例
├── .gitignore                         # Git忽略文件
├── CLAUDE.md                          # Claude Code配置
├── pyproject.toml                     # 项目配置
└── README.md                          # 项目说明
```

## 学习路径

### Step 1: 理解基础概念
- 什么是Agent（智能体）
- LangGraph的核心组件：State、Node、Edge
- 如何构建一个简单的对话Agent

### Step 2: 理解多节点工作流
- 多个节点如何组成工作流
- 节点之间的数据传递
- 简单的流水线架构

### Step 3: 真正的Multi-Agent系统⭐
- **区分概念**：多节点 vs Multi-Agent
- **Supervisor模式**：协调者 + 多个专业Agent
- **动态路由**：根据任务选择合适的Agent
- **独立决策**：每个Agent都有独立的能力和系统提示
- **多轮协作**：复杂任务的分解和协作

### Step 4: 工具调用（Tool Calling）⭐
- Agent如何使用工具
- 如何定义和绑定工具
- 工具执行的流程控制
- 扩展Agent的能力边界

### Step 5: 更多高级特性（规划中）
- 人机交互（Human-in-the-loop）
- 持久化和检查点（Checkpointing）
- 条件分支和复杂路由
- Agent间的直接通信

## 核心概念

### 什么是Multi-Agent？

**多节点（Multi-Node）≠ Multi-Agent**

- **多节点工作流**：多个节点按固定流程执行，类似流水线
- **Multi-Agent系统**：多个独立的智能体，能够自主决策和协作

真正的Multi-Agent系统特征：
1. **独立Agent**：每个Agent有独立的LLM实例、系统提示和能力
2. **动态路由**：根据任务动态选择调用哪个Agent
3. **自主决策**：Agent可以决定是否使用工具、如何响应
4. **协作机制**：通过Supervisor协调或Agent间直接通信

### State（状态）
State是Agent系统的数据容器，存储了整个执行过程中的所有信息。

### Node（节点）
Node是执行具体任务的单元，每个Agent通常对应一个Node。

### Edge（边）
Edge定义了Node之间的连接关系，决定了执行流程。可以是：
- **静态边**：固定的连接关系
- **条件边**：根据状态动态选择路径

### Graph（图）
Graph是整个Agent系统的结构，由多个Node和Edge组成。

### Supervisor（协调者）
在Multi-Agent系统中，Supervisor负责：
- 理解用户任务
- 选择合适的专业Agent
- 协调多个Agent的协作
- 决定任务是否完成

### Tool（工具）
Tool是Agent执行实际操作的能力扩展，如：
- 调用API
- 执行计算
- 搜索信息
- 操作数据库

## 常见问题

### Q: 多节点和Multi-Agent有什么区别？
A:
- **多节点**：多个节点按固定顺序执行，如 A → B → C 的流水线
- **Multi-Agent**：有协调者(Supervisor)根据任务动态选择调用哪个Agent，Agent之间可以多轮协作

### Q: 什么时候用多节点，什么时候用Multi-Agent？
A:
- **多节点适用**：流程固定、步骤明确的任务（如：数据处理流水线）
- **Multi-Agent适用**：需要根据任务类型动态选择处理方式的场景（如：智能客服、复杂任务规划）

### Q: 为什么选择LangGraph？
A: LangGraph是LangChain推出的专门用于构建Agent系统的框架，提供了：
- 清晰的State、Node、Edge抽象
- 强大的条件路由能力
- 内置的工具调用支持
- 检查点和持久化功能

### Q: 需要什么基础知识？
A: 基本的Python编程知识和对LLM的基础了解即可。项目中的每个示例都有详细注释，适合小白学习。

### Q: 如何自定义Agent？
A: 参考examples目录下的示例代码，你可以：
1. 修改System Prompt定义Agent的角色和能力
2. 添加新的工具扩展Agent的功能
3. 调整路由逻辑改变Agent的协作方式

### Q: 示例3和示例4有什么区别？
A:
- **示例3**：纯对话型Multi-Agent，Agent只通过LLM生成文本
- **示例4**：带工具的Multi-Agent，Agent可以调用工具执行实际操作（计算、搜索等）

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License
