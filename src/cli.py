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
    no_args_is_help=True,
)


def _mask_key(key: str) -> str:
    if not key:
        return "未配置"
    if len(key) <= 8:
        return "已配置 (******)"
    return f"已配置 ({key[:3]}...{key[-4:]})"


async def _run_write_async(
    topic: str,
    files: list[str],
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
            f"📚 [yellow]参考资料[/yellow]: {files or '无本地文件（将执行网络检索）'}\n"
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

    # 懒加载本地模型，保持 CLI 启动极速
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
        console.print("[bold green]✓ BAAI/bge-reranker-v2-m3 重排模型已就绪 (MPS 加速)[/bold green]")

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
        local_files=files,
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
    help="🚀 启动技术文章多 Agent 协同写作流程",
)
def write(
    topic: Annotated[
        str,
        typer.Option(
            "-t",
            "--topic",
            help="文章核心技术主题（例如：'大模型 KV-Cache 显存优化技术演进'）",
            prompt="请输入要写作的技术主题",
        ),
    ],
    files: Annotated[
        Optional[list[str]],
        typer.Option(
            "-f",
            "--file",
            help="本地参考文档/论文路径（支持 .pdf, .md, .txt，可多次指定）",
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
            help="LLM 模型名称（默认读取环境变量 LLM_MODEL 或 gpt-4o）",
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
        bool,
        typer.Option(
            "--rerank/--no-rerank",
            help="是否启用 BAAI/bge-reranker-v2-m3 本地深度精排",
        ),
    ] = False,
    output_dir: Annotated[
        str,
        typer.Option(
            "-o",
            "--output-dir",
            help="生成文章的输出保存目录",
        ),
    ] = "output",
):
    asyncio.run(
        _run_write_async(
            topic=topic,
            files=files or [],
            instructions=instructions,
            model=model,
            base_url=base_url,
            api_key=api_key,
            rerank=rerank,
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
        "[green]✓ 已配置[/green]"
        if os.getenv("LLM_MODEL")
        else "[cyan]• 默认[/cyan]"
    )
    table.add_row("LLM 模型", model_status, config.model)

    base_url_status = (
        "[green]✓ 已配置[/green]" if config.base_url else "[cyan]• 默认[/cyan]"
    )
    base_url_desc = config.base_url or "https://api.openai.com/v1 (官方)"
    table.add_row("LLM Base URL", base_url_status, base_url_desc)

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
            f"🔹 [bold]TAVILY_API_KEY[/bold]: {_mask_key(config.tavily_api_key)}",
            title="[bold cyan]zylo 当前有效配置[/bold cyan]",
            border_style="cyan",
        )
    )


def main():
    app()


if __name__ == "__main__":
    main()
