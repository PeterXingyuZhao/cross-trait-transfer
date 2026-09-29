import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from loguru import logger

def run_visible(command, log_path):
    """Stream child output into Colab and retain the full failure log."""
    from collections import deque
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=30)
    logger.info("Running {}", " ".join(map(str, command)))
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                tail.append(line.rstrip())
                logger.info("{}", line.rstrip())
            code = process.wait()
        except BaseException:
            import signal
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    if code:
        raise RuntimeError(f"Command failed (exit {code}); log: {log_path}\n" + "\n".join(tail))


def python_version(python):
    """Read an environment's interpreter version without importing GPU packages."""
    if not python.is_file():
        return None
    try:
        result = subprocess.run([str(python), "-I", "-c",
                                 "import sys; sys.stdout.write(str(sys.version_info.major)+'.'+str(sys.version_info.minor))"],
                                capture_output=True, text=True, timeout=20)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def has_pip(python):
    """Check that an existing environment can install its pinned dependencies."""
    try:
        result = subprocess.run([str(python), "-m", "pip", "--version"],
                                capture_output=True, text=True, timeout=20)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def unfinished_science(scratch, name):
    """Do not replace the interpreter of an unfinished condition."""
    category = {"venv-generation": "datasets", "venv-training": "training"}.get(name)
    if category is None:
        return False
    root = scratch / "work" / category
    marker_name = "generation_manifest.json" if category == "datasets" else "manifest.json"
    for parent in root.iterdir() if root.is_dir() else []:
        if parent.is_dir() and any(parent.iterdir()):
            marker = parent / marker_name
            if not marker.is_file() or json.loads(marker.read_text()).get("status") != "complete":
                return True
    return False


def prepare_python312(scratch):
    """Create Python 3.12 environments even when Colab's kernel is Python 3.13."""
    logs = scratch / "bootstrap_logs"
    run_visible([sys.executable, "-m", "pip", "install", "--quiet", "uv==0.8.22"], logs / "install_uv.log")
    uv = [sys.executable, "-m", "uv"]
    run_visible([*uv, "python", "install", "3.12"], logs / "install_python312.log")
    for name in ("venv-bootstrap", "venv-generation", "venv-training"):
        environment = scratch / name
        if environment.is_symlink():
            raise ValueError(f"Environment directory cannot be a symlink: {environment}")
        python = environment / "bin/python"
        version = python_version(python)
        if version != "3.12" or not has_pip(python):
            if unfinished_science(scratch, name):
                raise RuntimeError(f"Preserve the original Python environment for unfinished work: {name}")
            if environment.exists():
                # Preserve a failed/incompatible environment; never touch work or checkpoints.
                backup = Path(tempfile.mkdtemp(prefix=name + "-previous-", dir=scratch)) / "environment"
                environment.rename(backup)
                logger.info("Preserved incompatible environment at {}", backup)
            run_visible([*uv, "venv", "--python", "3.12", "--managed-python", "--seed", str(environment)],
                        logs / (name + ".log"))
        if python_version(python) != "3.12":
            raise RuntimeError(f"Failed to create Python 3.12 environment: {environment}")
        # uv seeds pip without depending on the system's python3-venv/ensurepip package.
        run_visible([str(python), "-m", "pip", "--version"], logs / (name + "-pip.log"))
    runner_python = scratch / "venv-bootstrap/bin/python"
    run_visible([str(runner_python), "-m", "pip", "install", "--quiet", "loguru"],
                logs / "bootstrap_loguru.log")
    return runner_python
