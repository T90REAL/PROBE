import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Mapping

from probe.tools.registry import ToolRegistry


def register_bash_tool(registry: ToolRegistry, *, timeout: int = 120) -> None:
    @registry.tool(
        name="bash",
        description="Run a bash command in the repository workspace.",
        params={
            "command": {
                "type": "string",
                "description": "The bash command to run. Multi-line scripts are allowed.",
            }
        },
    )
    def bash(ctx: Any, *, command: str) -> dict[str, Any]:
        return run_bash_command(command, cwd=_repo_root(ctx), timeout=timeout)


def run_bash_command(command: str, *, cwd: str | Path = ".", timeout: int = 120) -> dict[str, Any]:
    try:
        proc = _run(command, str(Path(cwd)), os.environ.copy(), timeout)
        return {
            "output": proc.stdout,
            "returncode": proc.returncode,
            "exception_info": "",
        }
    except Exception as exc:
        raw_output = getattr(exc, "output", None)
        output = raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
        return {
            "output": output,
            "returncode": -1,
            "exception_info": f"An error occurred while executing the command: {exc}",
            "extra": {"exception_type": type(exc).__name__, "exception": str(exc)},
        }


def _run(command: str, cwd: str, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        shell=True,
        text=True,
        cwd=cwd,
        env=env,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, _ = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout)
    return subprocess.CompletedProcess(command, process.returncode, stdout=stdout)


def _repo_root(ctx: Any) -> str:
    if isinstance(ctx, Mapping):
        return str(ctx.get("repo_root") or ".")
    return str(getattr(ctx, "repo_root", "."))
