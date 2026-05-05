# agent.py — ForgeCode 核心 Agent Loop

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Optional, Protocol


# ────────────────────────────────
# 基础数据结构
# ────────────────────────────────

@dataclass
class ToolCall:
    """模型请求调用的工具"""
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolResult:
    """工具执行后的结果"""
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class AgentMessage:
    """Agent 内部消息表示"""
    role: str  # "user" | "assistant" | "tool"
    content: Any


@dataclass
class Usage:
    """Token 使用量"""
    input_tokens: int
    output_tokens: int


@dataclass
class LLMResponse:
    """LLM 流式聚合后的响应"""
    content_blocks: List[Any] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(0, 0))
    stop_reason: Optional[str] = None


# ────────────────────────────────
# 协议 / 抽象接口
# ────────────────────────────────

class LLMClient(Protocol):
    """LLM 客户端协议——可替换为任意后端(Anthropic/OpenAI/本地)"""

    async def stream_chat(
        self,
        messages: List[AgentMessage],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Any]:
        """流式返回 LLM 的 content blocks"""
        ...


class ToolExecutor(Protocol):
    """工具执行器协议"""

    async def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """执行工具并返回字符串结果"""
        ...


class PermissionChecker(Protocol):
    """权限检查器协议"""

    def check(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        返回 {"action": "allow"|"deny"|"confirm", "message": str}
        """
        ...


class EventEmitter:
    """简单事件发射器，用于循环内外通信"""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable[..., Any]]] = {}

    def on(self, event: str, handler: Callable[..., Any]) -> None:
        self._handlers.setdefault(event, []).append(handler)

    async def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for handler in self._handlers.get(event, []):
            if asyncio.iscoroutinefunction(handler):
                await handler(*args, **kwargs)
            else:
                handler(*args, **kwargs)


# ────────────────────────────────
# 核心：Agent Loop
# ────────────────────────────────

class AgentLoop:
    """
    ForgeCode 最内层 Agent Loop

    职责：
    1. 接收用户消息 → 追加到对话历史
    2. while True 循环：
       a. 压缩上下文（如果需要）
       b. 流式调用 LLM，边收边解析 tool_use
       c. 并发/串行执行工具
       d. 将 tool_result 注入历史
       e. 如果模型返回纯文本（无 tool_use）→ break
    3. 错误恢复（截断、API 错误、上下文超限）
    4. 支持中断（abort）
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        permission_checker: PermissionChecker,
        tools_schema: List[Dict[str, Any]],
        max_iterations: int = 100,
        max_context_tokens: int = 200_000,
        compression_threshold: float = 0.935,
    ) -> None:
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.permission_checker = permission_checker
        self.tools_schema = tools_schema
        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens
        self.compression_threshold = compression_threshold

        self.messages: List[AgentMessage] = []
        self.events = EventEmitter()

        # 运行状态
        self._aborted = False
        self._iteration = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # 用户已确认的危险操作缓存
        self._confirmed_paths: set = set()

    # ── 公共控制 ──

    def abort(self) -> None:
        """外部信号：中断当前循环"""
        self._aborted = True

    def reset(self) -> None:
        """重置状态，开始新对话"""
        self.messages.clear()
        self._aborted = False
        self._iteration = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._confirmed_paths.clear()

    # ── 核心入口 ──

    async def run(self, user_message: str) -> AsyncIterator[str]:
        """
        启动 Agent Loop，以异步生成器形式产出：
        - 模型生成的文本片段（实时流式）
        - 工具调用信息
        - 最终结果
        """
        self.messages.append(AgentMessage(role="user", content=user_message))
        await self.events.emit("user_message", user_message)

        while True:
            if self._aborted:
                await self.events.emit("aborted")
                yield "[Agent loop aborted by user]"
                break

            self._iteration += 1
            if self._iteration > self.max_iterations:
                yield "[Max iterations reached]"
                break

            # ── 1. 上下文压缩 ──
            await self._maybe_compress_context()

            # ── 2. 流式调用 LLM ──
            try:
                llm_response = await self._stream_llm()
            except Exception as exc:
                # 错误恢复：API 异常 → 重试或降级
                recovered = await self._handle_llm_error(exc)
                if recovered:
                    continue
                yield f"[LLM Error: {exc}]"
                break

            self.total_input_tokens += llm_response.usage.input_tokens
            self.total_output_tokens += llm_response.usage.output_tokens

            # ── 3. 解析模型输出 ──
            text_blocks, tool_calls = self._parse_response(llm_response)

            # 将 assistant 消息写入历史（包含 text + tool_use）
            self.messages.append(
                AgentMessage(role="assistant", content=llm_response.content_blocks)
            )

            # 产出文本（实时展示给用户）
            for text in text_blocks:
                yield text

            # ── 4. 无工具调用 → 任务完成 ──
            if not tool_calls:
                await self.events.emit("turn_complete", llm_response)
                break

            # ── 5. 执行工具 ──
            tool_results = await self._execute_tools(tool_calls)

            # 将 tool_results 注入历史
            self.messages.append(
                AgentMessage(role="user", content=[
                    {"type": "tool_result", "tool_use_id": tr.tool_use_id,
                     "content": tr.content, "is_error": tr.is_error}
                    for tr in tool_results
                ])
            )

            await self.events.emit("tool_results", tool_results)

        await self.events.emit("loop_complete", self.total_input_tokens, self.total_output_tokens)

    # ── 内部方法 ──

    async def _stream_llm(self) -> LLMResponse:
        """
        流式调用 LLM，边收边聚合。
        参考 Claude Code：在流式过程中即可开始解析 tool_use，
        但这里先完整聚合后再返回（简化版）。
        """
        response = LLMResponse()
        content_blocks: List[Any] = []

        async for chunk in self.llm_client.stream_chat(self.messages, self.tools_schema):
            # chunk 假设为 dict: {"type": "text", "text": "..."}
            # 或 {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
            content_blocks.append(chunk)

        response.content_blocks = content_blocks
        # usage 在真实场景中从响应头或最后一块获取
        response.usage = Usage(input_tokens=0, output_tokens=0)
        return response

    def _parse_response(self, response: LLMResponse) -> tuple[List[str], List[ToolCall]]:
        """将 LLMResponse 拆分为文本块和工具调用列表"""
        texts: List[str] = []
        tools: List[ToolCall] = []

        for block in response.content_blocks:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tools.append(ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        input=block.get("input", {}),
                    ))
        return texts, tools

    async def _execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        执行工具调用列表。
        只读工具可并行；写操作串行（简化版全部串行，可扩展为并行）。
        """
        results: List[ToolResult] = []

        for tc in tool_calls:
            if self._aborted:
                break

            await self.events.emit("tool_call_start", tc.name, tc.input)

            # 权限检查
            perm = self.permission_checker.check(tc.name, tc.input)

            if perm.get("action") == "deny":
                results.append(ToolResult(
                    tool_use_id=tc.id,
                    content=f"Action denied: {perm.get('message', '')}",
                    is_error=True,
                ))
                continue

            if perm.get("action") == "confirm" and perm.get("message"):
                if perm["message"] not in self._confirmed_paths:
                    confirmed = await self._confirm_dangerous(perm["message"])
                    if not confirmed:
                        results.append(ToolResult(
                            tool_use_id=tc.id,
                            content="User denied this action.",
                            is_error=True,
                        ))
                        continue
                    self._confirmed_paths.add(perm["message"])

            # 实际执行
            try:
                output = await self.tool_executor.execute(tc.name, tc.input)
                results.append(ToolResult(tool_use_id=tc.id, content=output))
                await self.events.emit("tool_call_end", tc.name, output)
            except Exception as exc:
                results.append(ToolResult(
                    tool_use_id=tc.id,
                    content=f"Tool execution error: {exc}",
                    is_error=True,
                ))
                await self.events.emit("tool_call_error", tc.name, exc)

        return results

    async def _confirm_dangerous(self, message: str) -> bool:
        """
        危险操作确认。真实场景中通过 UI/CLI 询问用户。
        这里提供一个可覆盖的钩子。
        """
        await self.events.emit("confirm_required", message)
        # 默认实现：自动允许（子 Agent 或测试场景）
        # 实际应由外部注入回调
        return True

    async def _maybe_compress_context(self) -> None:
        """
        上下文压缩流水线。
        当 token 数接近阈值时，从轻量到激进逐级压缩。
        """
        # 简化：计算估算 token（真实场景用 tiktoken/anthropic tokenizer）
        estimated_tokens = self._estimate_tokens()
        if estimated_tokens < self.max_context_tokens * self.compression_threshold:
            return

        await self.events.emit("context_compress_start", estimated_tokens)

        # 1. Snip：将大块 tool output 替换为占位符
        self._snip_large_outputs()

        # 2. Microcompact：对 tool results 做局部摘要
        self._microcompact()

        # 3. Context Collapse：将早期消息折叠为摘要
        self._collapse_context()

        # 4. Autocompact：全量摘要（最激进）
        if self._estimate_tokens() >= self.max_context_tokens * self.compression_threshold:
            await self._autocompact()

        await self.events.emit("context_compress_end", self._estimate_tokens())

    def _estimate_tokens(self) -> int:
        """粗略估算当前消息历史的 token 数"""
        # 简化：每消息 100 token + 内容长度/4
        total = 0
        for msg in self.messages:
            total += 100
            content = msg.content
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                total += sum(len(str(item)) for item in content) // 4
        return total

    def _snip_large_outputs(self) -> None:
        """剪裁过大的工具输出"""
        SNIP_THRESHOLD = 2000
        for msg in self.messages:
            if msg.role == "user" and isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        content = item.get("content", "")
                        if isinstance(content, str) and len(content) > SNIP_THRESHOLD:
                            item["content"] = (
                                content[:SNIP_THRESHOLD // 2]
                                + f"\n... [{len(content)} chars snipped] ...\n"
                                + content[-SNIP_THRESHOLD // 2:]
                            )

    def _microcompact(self) -> None:
        """对工具结果做局部压缩（简化版）"""
        pass  # 可扩展：调用小型模型对长输出做摘要

    def _collapse_context(self) -> None:
        """上下文折叠：将早期对话折叠为系统摘要"""
        # 保留最近 10 条消息，将更早的折叠
        KEEP_RECENT = 10
        if len(self.messages) <= KEEP_RECENT + 1:
            return

        old_messages = self.messages[:-KEEP_RECENT]
        recent_messages = self.messages[-KEEP_RECENT:]

        # 生成摘要（简化：直接计数）
        summary = f"[Context collapsed: {len(old_messages)} earlier messages summarized]"
        self.messages = [
            AgentMessage(role="system", content=summary),
            *recent_messages,
        ]

    async def _autocompact(self) -> None:
        """全量自动压缩：调用 LLM 生成整个历史的摘要"""
        # 真实场景中：用低成本模型对历史做摘要
        summary_msg = AgentMessage(
            role="system",
            content="[Autocompact: Full conversation summarized due to context limit]",
        )
        # 保留系统消息和最后一条用户消息
        self.messages = [
            summary_msg,
            self.messages[-1],
        ]

    async def _handle_llm_error(self, exc: Exception) -> bool:
        """
        错误恢复：API 错误、截断、上下文超限等。
        返回 True 表示已恢复，可继续循环；False 表示无法恢复。
        """
        # 可按错误类型分发处理
        error_str = str(exc).lower()

        if "context" in error_str or "token" in error_str:
            await self._autocompact()
            return True

        if "rate limit" in error_str:
            await asyncio.sleep(2)
            return True

        # 其他错误：不可恢复
        return False


# ────────────────────────────────
# 示例：简单 CLI 工具执行器
# ────────────────────────────────

class SimpleToolExecutor:
    """示例工具执行器——可替换为真实实现"""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Coroutine[Any, Any, str]]] = {}

    def register(self, name: str, fn: Callable[..., Coroutine[Any, Any, str]]) -> None:
        self._tools[name] = fn

    async def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return await self._tools[name](**arguments)


# ── 占位函数（保持与旧代码兼容）──

def print_cost(input_tokens: int, output_tokens: int) -> None:
    pass

def print_tool_call(name: str, inp: Dict[str, Any]) -> None:
    pass

def print_tool_result(name: str, result: str) -> None:
    pass

def check_permission(
    name: str, inp: Dict[str, Any], mode: str, plan_path: Optional[str]
) -> Dict[str, Any]:
    return {"action": "allow"}


class PermissiveChecker:
    """默认允许所有操作的权限检查器"""

    def check(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "allow"}
