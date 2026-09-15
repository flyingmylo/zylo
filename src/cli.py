import asyncio
import os
import sys
from typing import Annotated, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.llm.config import LLMConfig
from src.llm.openai_provider import OpenAICompatibleProvider
from src.orchestrator import WritingOrchestrator
from src.state import WritingState

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


@app.command(
    name="write",
    help="🚀 启动技术文章多 Agent 协同写作流程（支持位置参数直传或问答向导）",
)
def write(
    inputs: Annotated[
        Optional[list[str]],
        typer.Argument(
            help="文章主题；后续参数可直接跟本地文件（.pdf/.md/.txt）或论文 URL（如 arXiv 链接）",
        ),
    ] = None,
    topic: Annotated[
        Optional[str],
        typer.Option(
            "-t",
            "--topic",
            help="文章核心技术主题（亦可直接作为第一个位置参数传参）",
        ),
    ] = None,
    files: Annotated[
        Optional[list[str]],
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
        Optional[str],
        typer.Option(
            "-m",
            "--model",
            help="LLM 模型名称（默认读取环境变量 LLM_MODEL 或根据端点自动推导）",
        ),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option(
            "--base-url",
            help="LLM API Base URL（支持 DeepSeek, 通义千问, 智谱等兼容端点）",
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            help="LLM API Key（默认从环境变量 LLM_API_KEY 读取）",
        ),
    ] = None,
    rerank: Annotated[
        Optional[bool],
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
    model_status = (
        "[green]✓ 已配置[/green]" if os.getenv("LLM_MODEL") else "[cyan]• 默认[/cyan]"
    )
    table.add_row("LLM 模型", model_status, config.model)

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
            f"🔹 [bold]LLM_MODEL[/bold]: {config.model}\n"
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
    subcommands = {"check", "config", "write"}
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
