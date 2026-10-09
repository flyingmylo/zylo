import asyncio
import sys
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.embeddings.base import EmbeddingProvider
from src.llm.config import LLMConfig
from src.llm.openai_provider import OpenAICompatibleProvider
from src.orchestrator import WritingOrchestrator
from src.runs import RunStatus
from src.state import WritingState, serialize_state

console = Console()
app = typer.Typer(
    name="zylo",
    help="🪶 [bold cyan]zylo[/bold cyan] - 基于纯手写多智能体架构的中文深度技术博客与长文写作系统",
    rich_markup_mode="rich",
)


def _mask_key(key: str) -> str:
    if not key:
        return "未配置"
    if len(key) <= 8:
        return "已配置 (******)"
    return f"已配置 ({key[:3]}...{key[-4:]})"


async def _run_write_async(
    topic: str,
    sources: list[str],
    instructions: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    rerank: bool,
    output_dir: str,
):
    config = LLMConfig()

    effective_api_key = api_key or config.api_key
    effective_base_url = base_url or config.base_url
    effective_model = model or config.model

    if not effective_api_key:
        console.print("[bold red]❌ 错误：未检测到 LLM API Key！[/bold red]")
        console.print(
            "请通过环境变量 [cyan]LLM_API_KEY[/cyan] 或参数 [cyan]--api-key[/cyan] 指定。"
        )
        raise typer.Exit(code=1)

    if not effective_model:
        console.print("[bold red]❌ 错误：未检测到模型名称！[/bold red]")
        console.print(
            "本项目不提供默认模型，请通过环境变量 [cyan]LLM_MODEL[/cyan] "
            "或参数 [cyan]--model[/cyan] 显式指定。"
        )
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold cyan]🪶 zylo 写作任务启动[/bold cyan]\n"
            f"🎯 [yellow]技术主题[/yellow]: {topic}\n"
            f"🤖 [yellow]模型配置[/yellow]: {effective_model} (Base URL: {effective_base_url or 'OpenAI 官方'})\n"
            f"📚 [yellow]参考资料[/yellow]: {sources or '无本地/网页资料（将执行网络检索）'}\n"
            f"⚡ [yellow]重排精排[/yellow]: {'已开启 (BAAI/bge-reranker-v2-m3)' if rerank else '未开启 (默认相对 Top-K)'}\n"
            f"📁 [yellow]输出目录[/yellow]: {output_dir}",
            title="[bold green]任务配置面板[/bold green]",
            border_style="cyan",
        )
    )

    llm_provider = OpenAICompatibleProvider(
        api_key=effective_api_key,
        base_url=effective_base_url,
        model=effective_model,
    )

    # 懒加载本地模型，放开原生进度条显示
    console.print(
        "[bold cyan]🔹 正在载入 BAAI/bge-m3 嵌入模型 (首次运行将通过镜像源自动下载权重)...[/bold cyan]"
    )
    from src.embeddings.bge_provider import BGEM3EmbeddingProvider

    embedding_provider = BGEM3EmbeddingProvider()
    console.print("[bold green]✓ BAAI/bge-m3 嵌入模型已就绪 (MPS 加速)[/bold green]")

    reranker_provider = None
    if rerank:
        console.print(
            "[bold cyan]🔹 正在载入 BAAI/bge-reranker-v2-m3 重排模型 (首次运行将通过镜像源自动下载权重)...[/bold cyan]"
        )
        from src.embeddings.bge_reranker import BGERerankerProvider

        reranker_provider = BGERerankerProvider()
        console.print(
            "[bold green]✓ BAAI/bge-reranker-v2-m3 重排模型已就绪 (MPS 加速)[/bold green]"
        )

    def progress_logger(message: str, state: WritingState):
        console.print(f"[dim]{message}[/dim]")

    orchestrator = WritingOrchestrator(
        llm=llm_provider,
        embedding_provider=embedding_provider,
        reranker_provider=reranker_provider,
        tavily_api_key=config.tavily_api_key,
        progress_callback=progress_logger,
    )

    state = await orchestrator.execute(
        topic=topic,
        local_files=sources,
        extra_instructions=instructions,
        output_dir=output_dir,
    )

    console.print("\n" + "=" * 60)
    console.print(
        Panel(
            f"[bold green]✨ 文章生成完毕，已成功归档！[/bold green]\n\n"
            f"📌 [bold]最终标题[/bold]: {state.outline_title}\n"
            f"📝 [bold]正文字数[/bold]: 约 {len(state.full_draft)} 字\n"
            f"🏅 [bold]质检评分[/bold]: [bold yellow]{state.review_score:.1f}[/bold yellow] / 100\n"
            f"🔄 [bold]反思修订[/bold]: {state.revision_count} 轮\n"
            f"📁 [bold]归档路径[/bold]: 请查看 [cyan]{output_dir}/[/cyan] 目录",
            title="[bold green]生成成功[/bold green]",
            border_style="green",
        )
    )

    if state.errors:
        console.print(
            f"[yellow]⚠ 任务完成，但有 {len(state.errors)} 条非致命错误：[/yellow]"
        )
        for err in state.errors:
            console.print(f"  [dim]- {err}[/dim]")


@app.command(
    name="write",
    help="🚀 启动技术文章多 Agent 协同写作流程（支持位置参数直传或问答向导）",
)
def write(
    inputs: Annotated[
        list[str] | None,
        typer.Argument(
            help="文章主题；后续参数可直接跟本地文件（.pdf/.md/.txt）或论文 URL（如 arXiv 链接）",
        ),
    ] = None,
    topic: Annotated[
        str | None,
        typer.Option(
            "-t",
            "--topic",
            help="文章核心技术主题（亦可直接作为第一个位置参数传参）",
        ),
    ] = None,
    files: Annotated[
        list[str] | None,
        typer.Option(
            "-f",
            "--file",
            help="参考文档路径或网页 URL，可多次指定（支持智能嗅探）",
        ),
    ] = None,
    instructions: Annotated[
        str,
        typer.Option(
            "-i",
            "--instructions",
            help="给 Agent 的额外写作要求或目标读者定位",
        ),
    ] = "",
    model: Annotated[
        str | None,
        typer.Option(
            "-m",
            "--model",
            help="LLM 模型名称（默认读取环境变量 LLM_MODEL 或根据端点自动推导）",
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="LLM API Base URL（支持 DeepSeek, 通义千问, 智谱等兼容端点）",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="LLM API Key（默认从环境变量 LLM_API_KEY 读取）",
        ),
    ] = None,
    rerank: Annotated[
        bool | None,
        typer.Option(
            "--rerank/--no-rerank",
            help="是否启用 BAAI/bge-reranker-v2-m3 本地深度精排（默认: auto 智能条件激活）",
        ),
    ] = None,
    output_dir: Annotated[
        str,
        typer.Option(
            "-o",
            "--output-dir",
            help="生成文章的输出保存目录",
        ),
    ] = "output",
):
    all_sources = list(files or [])
    final_topic = topic

    # 1. 智能位置嗅探与合并 (Q3-A)
    if inputs:
        if not final_topic:
            final_topic = inputs[0].strip()
            remaining = inputs[1:]
        else:
            remaining = inputs

        for item in remaining:
            item_str = item.strip()
            if item_str:
                all_sources.append(item_str)

    # 2. 交互式两步向导模式 (Q1-C & Q6-A)
    if not final_topic:
        console.print("\n[bold cyan]🪶 欢迎使用 zylo 智能写作向导[/bold cyan]\n")
        final_topic = typer.prompt("📌 请输入文章主题").strip()
        source_in = typer.prompt(
            "📚 参考文件或论文链接 [直接回车跳过]", default=""
        ).strip()
        if source_in:
            all_sources.append(source_in)

    # 3. 三态 Reranker 决策策略 (Q4-A + auto 智能激活)
    config = LLMConfig()
    if rerank is not None:
        effective_rerank = rerank
    else:
        policy = config.enable_rerank.lower()
        if policy in ("true", "1", "yes", "on"):
            effective_rerank = True
        elif policy in ("false", "0", "no", "off"):
            effective_rerank = False
        else:
            # auto / default 模式：存在本地文档或论文链接时智能激活
            effective_rerank = bool(all_sources)

    asyncio.run(
        _run_write_async(
            topic=final_topic,
            sources=all_sources,
            instructions=instructions,
            model=model,
            base_url=base_url,
            api_key=api_key,
            rerank=effective_rerank,
            output_dir=output_dir,
        )
    )


@app.command(
    name="serve",
    help="🌐 启动 API 服务（--mock 为离线演示模式，无需任何 API Key 与本地模型）",
)
def serve(
    mock: Annotated[
        bool,
        typer.Option(
            "--mock",
            help="离线演示模式：Mock LLM + 固定向量嵌入，不访问网络不产生费用",
        ),
    ] = False,
    host: Annotated[str, typer.Option("--host", help="监听地址")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="监听端口")] = 8000,
):
    # 装配放函数内：不把 fastapi/uvicorn 的导入成本强加给 write/check 等命令
    import uvicorn

    from api.app import create_app
    from api.bus import TraceBus
    from api.runner import JobRunner
    from api.store import RunStore

    config = LLMConfig()
    # 嵌入模型权重 2GB：真实模式下进程内只加载一次，跨 run 复用
    shared: dict[str, EmbeddingProvider] = {}

    def make_executor(run) -> WritingOrchestrator:
        if mock:
            from src.embeddings.dummy import DummyEmbeddingProvider
            from src.llm.mock import MockLLMProvider

            # 每次运行一个新 Mock 实例：审稿轮次状态按 run 隔离
            return WritingOrchestrator(
                llm=MockLLMProvider(),
                embedding_provider=DummyEmbeddingProvider(),
                kb_persist_dir=f"data/chroma/{run.id}",
            )

        if not config.api_key or not config.model:
            console.print(
                "[bold red]❌ 真实模式需要 LLM_API_KEY 与 LLM_MODEL；"
                "离线体验请加 --mock[/bold red]"
            )
            raise typer.Exit(code=1)

        if "embedding" not in shared:
            from src.embeddings.bge_provider import BGEM3EmbeddingProvider

            console.print("[bold cyan]🔹 正在载入 BAAI/bge-m3 嵌入模型...[/bold cyan]")
            shared["embedding"] = BGEM3EmbeddingProvider()

        return WritingOrchestrator(
            llm=OpenAICompatibleProvider(
                api_key=config.api_key, model=config.model, base_url=config.base_url
            ),
            embedding_provider=shared["embedding"],
            tavily_api_key=config.tavily_api_key or None,
            kb_persist_dir=f"data/chroma/{run.id}",
        )

    # SQLite 事实来源：重启后运行列表、详情与事件历史仍可查询
    store = RunStore("data/zylo.db")

    # 启动恢复：上一个进程遗留的 RUNNING 已无宿主任务，落位 PARTIAL，
    # 用户可通过 zylo resume <run_id> 从最新快照续跑
    from api.runner import recover_stale_runs

    for stale in recover_stale_runs(store):
        console.print(
            f"[yellow]⚠ 检测到中断的运行 {stale.id}（{stale.topic}），"
            f"已转为 PARTIAL，可用 zylo resume {stale.id} 续跑[/yellow]"
        )

    bus = TraceBus()
    app = create_app(bus=bus, runner=JobRunner(bus, store), executor_factory=make_executor)

    mode_desc = "[green]Mock 离线演示[/green]" if mock else "[yellow]真实模型[/yellow]"
    console.print(
        Panel.fit(
            f"🪶 zylo API 服务\n"
            f"⚡ 模式: {mode_desc}\n"
            f"🌐 地址: [cyan]http://{host}:{port}[/cyan]\n"
            f"📡 事件流: [cyan]http://{host}:{port}/api/runs/{{run_id}}/events[/cyan]",
            title="[bold green]服务就绪[/bold green]",
            border_style="cyan",
        )
    )
    uvicorn.run(app, host=host, port=port)


@app.command(
    name="resume",
    help="▶️ 从最新阶段快照续跑中断的任务（zylo resume <run_id>）",
)
def resume_cmd(
    run_id: Annotated[str, typer.Argument(help="要恢复的运行 ID")],
    output_dir: Annotated[str, typer.Option("-o", "--output-dir")] = "output",
):
    import asyncio

    from api.store import RunStore
    from src.state import deserialize_state

    store = RunStore("data/zylo.db")
    run = store.get_run(run_id)
    if run is None:
        console.print(f"[bold red]❌ 运行 {run_id} 不存在（数据库 data/zylo.db）[/bold red]")
        raise typer.Exit(code=1)

    if run.status.value not in ("partial", "failed"):
        console.print(
            f"[bold red]❌ 仅 PARTIAL/FAILED 状态可恢复，当前为 {run.status.value}；"
            f"serve 启动时会把遗留的 RUNNING 自动转为 PARTIAL[/bold red]"
        )
        raise typer.Exit(code=1)

    snapshot = store.latest_state_snapshot(run_id)
    if snapshot is None:
        console.print("[bold red]❌ 该运行没有可恢复的阶段快照[/bold red]")
        raise typer.Exit(code=1)

    try:
        resume_state = deserialize_state(snapshot)
    except ValueError as exc:
        console.print(f"[bold red]❌ 快照不可用：{exc}[/bold red]")
        raise typer.Exit(code=1) from None

    config = LLMConfig()
    if not config.api_key or not config.model:
        console.print("[bold red]❌ 恢复需要 LLM_API_KEY 与 LLM_MODEL[/bold red]")
        raise typer.Exit(code=1)

    from src.embeddings.bge_provider import BGEM3EmbeddingProvider

    console.print("[bold cyan]🔹 正在载入 BAAI/bge-m3 嵌入模型...[/bold cyan]")
    embedding = BGEM3EmbeddingProvider()

    def on_checkpoint(state) -> None:
        store.save_state_snapshot(run_id, state.current_stage.value, serialize_state(state))

    def progress(message: str, state) -> None:
        console.print(f"[dim]{message}[/dim]")

    run.transition(RunStatus.RUNNING)
    store.upsert_run(run)

    async def _run() -> WritingState:
        return await WritingOrchestrator(
            llm=OpenAICompatibleProvider(
                api_key=config.api_key, model=config.model, base_url=config.base_url
            ),
            embedding_provider=embedding,
            tavily_api_key=config.tavily_api_key or None,
            progress_callback=progress,
            kb_persist_dir=f"data/chroma/{run.id}",
        ).execute(
            topic=run.topic,
            output_dir=output_dir,
            resume_state=resume_state,
            on_checkpoint=on_checkpoint,
        )

    console.print(f"[bold green]▶️ 从阶段 {resume_state.current_stage.value} 续跑 {run_id}[/bold green]")
    try:
        asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 -- 恢复入口边界：任何异常转 FAILED 并保留快照
        run.error = str(exc)
        run.transition(RunStatus.FAILED)
        store.upsert_run(run)
        console.print(f"[bold red]❌ 续跑失败：{exc}[/bold red]")
        raise typer.Exit(code=1) from None

    run.transition(RunStatus.COMPLETED)
    store.upsert_run(run)
    console.print("[bold green]✅ 续跑完成，文章已导出[/bold green]")


@app.command(
    name="check",
    help="🔍 诊断本机硬件加速 (Metal MPS) 与环境变量状态",
)
def check():
    import torch

    config = LLMConfig()

    table = Table(
        title="zylo 系统与环境诊断报告",
        box=box.HORIZONTALS,
        header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("检查项", style="bold", width=18)
    table.add_column("状态", width=10)
    table.add_column("详情说明", style="dim")

    # Python 版本
    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    table.add_row("Python 版本", "[green]✓ 正常[/green]", f"Python {py_ver}")

    # Apple Silicon MPS 加速
    mps_available = torch.backends.mps.is_available()
    if mps_available:
        table.add_row(
            "Apple Metal (MPS)",
            "[green]✓ 支持[/green]",
            "已检测到 Apple Silicon GPU 硬件加速就绪",
        )
    else:
        table.add_row(
            "Apple Metal (MPS)", "[yellow]! 不支持[/yellow]", "将回退到 CPU 运算"
        )

    # LLM API Key
    llm_status = "[green]✓ 已配置[/green]" if config.api_key else "[red]✗ 未配置[/red]"
    table.add_row("LLM API Key", llm_status, _mask_key(config.api_key))

    # LLM Model & Endpoint
    model_status = "[green]✓ 已配置[/green]" if config.model else "[red]✗ 未设置[/red]"
    table.add_row("LLM 模型", model_status, config.model or "未设置 (必须显式指定)")

    base_url_status = (
        "[green]✓ 已配置[/green]" if config.base_url else "[cyan]• 默认[/cyan]"
    )
    base_url_desc = config.base_url or "https://api.openai.com/v1 (官方)"
    table.add_row("LLM Base URL", base_url_status, base_url_desc)

    # Reranker 策略
    policy_str = config.enable_rerank.lower()
    if policy_str == "auto":
        rerank_desc = "auto (有参考文件/URL资料时智能激活)"
    elif policy_str in ("true", "1", "yes"):
        rerank_desc = "true (常态保持开启)"
    else:
        rerank_desc = "false (默认关闭)"
    table.add_row("重排 (Reranker)", "[green]✓ 已配置[/green]", rerank_desc)

    # Tavily 搜索
    tavily_status = (
        "[green]✓ 已配置[/green]"
        if config.tavily_api_key
        else "[yellow]! 未配置[/yellow]"
    )
    tavily_desc = (
        _mask_key(config.tavily_api_key)
        if config.tavily_api_key
        else "未配置 (在线搜索将跳过)"
    )
    table.add_row("Tavily 搜索 API", tavily_status, tavily_desc)

    console.print(table)


@app.command(
    name="config",
    help="⚙️ 显示当前激活的环境变量与模型配置",
)
def show_config():
    config = LLMConfig()
    console.print(
        Panel.fit(
            f"🔹 [bold]LLM_MODEL[/bold]: {config.model or '(未设置)'}\n"
            f"🔹 [bold]LLM_BASE_URL[/bold]: {config.base_url or '(默认 OpenAI 官方)'}\n"
            f"🔹 [bold]LLM_API_KEY[/bold]: {_mask_key(config.api_key)}\n"
            f"🔹 [bold]LLM_TEMPERATURE[/bold]: {config.temperature}\n"
            f"🔹 [bold]TAVILY_API_KEY[/bold]: {_mask_key(config.tavily_api_key)}\n"
            f"🔹 [bold]ENABLE_RERANK[/bold]: {config.enable_rerank}",
            title="[bold cyan]zylo 当前有效配置[/bold cyan]",
            border_style="cyan",
        )
    )


def main():
    # 智能快捷注入 (Q2-A)：若直接输入 zylo "主题" 或仅输入 zylo，自动映射为 write 命令
    subcommands = {"check", "config", "resume", "serve", "write"}
    raw_args = sys.argv[1:]
    if raw_args:
        first = raw_args[0]
        if (
            first not in ("--help", "-h", "--install-completion", "--show-completion")
            and first not in subcommands
        ):
            sys.argv.insert(1, "write")
    else:
        # 用户仅输入了 `zylo`，唤醒 write 交互式向导
        sys.argv.insert(1, "write")

    app()


if __name__ == "__main__":
    main()
