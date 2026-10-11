from database import (
    init_db,
    create_api_key,
    disable_api_key,
    list_api_keys,
    get_usage_summary,
)
from pathlib import Path
from typing import Annotated, Optional
import httpx
import shutil
import subprocess
import sys
import typer
import config

app = typer.Typer(help="aihost - your own OpenAI-compatible API for local models")


# create the API key in the DB and show it ONCE (only the hash is stored)
@app.command("create")
def get_api_name(api_key_name: Annotated[str, typer.Argument()]):
    """Create a new API key (shown once)."""
    api_key = create_api_key(name=api_key_name)
    typer.echo(f"created api key '{api_key_name}':\n\n    {api_key}\n")
    typer.echo("save it now - it can't be shown again.")


# disable an API key
@app.command("disable")
def disable_api(id: Annotated[int, typer.Argument()]):
    """Disable an API key by id."""
    try:
        changed = disable_api_key(id)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    if changed:
        typer.echo(f"disabled api key {id}")
    else:
        typer.echo(f"api key {id} was already disabled")


# list every api key
@app.command("list")
def list_keys():
    """List all API keys."""
    rows = list_api_keys()
    if not rows:
        typer.echo("no api keys yet - create one with: python cli.py create <name>")
        return
    typer.echo(f"{'ID':<5}{'NAME':<25}{'CREATED':<22}STATUS")
    for id, name, created_at, is_active in rows:
        status = "active" if is_active else "disabled"
        typer.echo(f"{id:<5}{name:<25}{created_at:<22}{status}")


# tokens / requests per key
@app.command("usage")
def usage():
    """Show requests and token usage per key."""
    rows = get_usage_summary()
    if not rows:
        typer.echo("no api keys yet")
        return
    typer.echo(
        f"{'ID':<5}{'NAME':<20}{'REQS':>7}{'ERRS':>7}{'PROMPT':>10}"
        f"{'COMPLETION':>12}{'TOTAL':>10}{'AVG MS':>10}"
    )
    for id, name, reqs, errs, prompt, completion, total, avg_ms in rows:
        typer.echo(
            f"{id:<5}{name:<20}{reqs:>7}{errs:>7}{prompt:>10}"
            f"{completion:>12}{total:>10}{avg_ms:>10.0f}"
        )


# start llama-server with a model (runs in the foreground)
@app.command("start-model")
def start_model(
    model: Annotated[Path, typer.Argument(help="path to a .gguf model file")],
    gpu_layers: Annotated[
        int, typer.Option(help="layers to offload to the GPU (999 = all, 0 = CPU only)")
    ] = 999,
    ctx_size: Annotated[int, typer.Option(help="context window in tokens")] = 4096,
    alias: Annotated[
        Optional[str], typer.Option(help="model name shown in /v1/models")
    ] = None,
    extra: Annotated[
        Optional[list[str]], typer.Option(help="extra flag passed to llama-server")
    ] = None,
):
    """Start llama-server with a .gguf model on the GPU."""
    if not model.is_file():
        typer.echo(f"model file not found: {model}", err=True)
        raise typer.Exit(code=1)
    if shutil.which(config.LLAMA_SERVER_BIN) is None:
        typer.echo(
            f"'{config.LLAMA_SERVER_BIN}' not found - install llama.cpp "
            "or set AIHOST_LLAMA_BIN",
            err=True,
        )
        raise typer.Exit(code=1)

    command = [
        config.LLAMA_SERVER_BIN,
        "--model", str(model),
        "--host", config.LLAMA_HOST,
        "--port", str(config.LLAMA_PORT),
        "--n-gpu-layers", str(gpu_layers),
        "--ctx-size", str(ctx_size),
        "--alias", alias or model.stem,
    ]  # fmt: skip
    command += extra or []
    typer.echo("running: " + " ".join(command))
    try:
        raise typer.Exit(code=subprocess.call(command))
    except KeyboardInterrupt:
        pass


# start the public API (this app)
@app.command("serve")
def serve(
    host: Annotated[str, typer.Option()] = config.API_HOST,
    port: Annotated[int, typer.Option()] = config.API_PORT,
    reload: Annotated[bool, typer.Option(help="auto reload on code change")] = False,
):
    """Start the public API (OpenAI-compatible, needs an API key)."""
    import uvicorn

    uvicorn.run("main:app", host=host, port=port, reload=reload)


# show what's currently running
@app.command("status")
def status():
    """Show whether the model server is running and which model."""
    try:
        health = httpx.get(f"{config.LLAMA_SERVER_URL}/health", timeout=5.0)
    except httpx.HTTPError:
        typer.echo(f"model server: not running ({config.LLAMA_SERVER_URL})")
        raise typer.Exit(code=1)

    if health.status_code != 200:
        typer.echo(f"model server: loading ({config.LLAMA_SERVER_URL})")
        return
    typer.echo(f"model server: running ({config.LLAMA_SERVER_URL})")
    try:
        models = httpx.get(f"{config.LLAMA_SERVER_URL}/v1/models", timeout=5.0).json()
        for m in models.get("data", []):
            typer.echo(f"  model: {m.get('id')}")
    except (httpx.HTTPError, ValueError):
        pass


# doctor to check everything
@app.command("doctor")
def doctor():
    """Check GPU, llama-server, database and model server."""
    ok = True

    def check(passed: bool, label: str, hint: str = ""):
        nonlocal ok
        mark = "ok  " if passed else "FAIL"
        typer.echo(f"[{mark}] {label}" + (f"  -> {hint}" if not passed and hint else ""))
        ok = ok and passed

    check(sys.version_info >= (3, 10), f"python {sys.version.split()[0]}", "need 3.10+")

    # GPU (optional - llama.cpp also runs on CPU, just slower)
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        gpus = result.stdout.strip().splitlines()
        check(bool(gpus), "gpu: " + (", ".join(gpus) or "none found"), "check drivers")
    elif sys.platform == "darwin":
        typer.echo("[ok  ] gpu: Apple Silicon (Metal)")
    else:
        typer.echo("[warn] gpu: nvidia-smi not found - models will run on CPU")

    check(
        shutil.which(config.LLAMA_SERVER_BIN) is not None,
        f"llama-server binary ({config.LLAMA_SERVER_BIN})",
        "install llama.cpp or set AIHOST_LLAMA_BIN",
    )

    try:
        init_db()
        check(True, f"database ({config.DB_PATH})")
        active = [row for row in list_api_keys() if row[3]]
        check(bool(active), f"{len(active)} active api key(s)", "python cli.py create <name>")
    except Exception as exc:
        check(False, f"database ({config.DB_PATH})", str(exc))

    try:
        response = httpx.get(f"{config.LLAMA_SERVER_URL}/health", timeout=5.0)
        check(
            response.status_code == 200,
            f"model server at {config.LLAMA_SERVER_URL}",
            "still loading the model",
        )
    except httpx.HTTPError:
        check(
            False,
            f"model server at {config.LLAMA_SERVER_URL}",
            "python cli.py start-model <model.gguf>",
        )

    raise typer.Exit(code=0 if ok else 1)


if __name__ == "__main__":
    init_db()
    app()
