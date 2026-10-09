"""Install only into the dedicated runtime; inherit the server's network route."""

import hashlib
import os
import shutil
import signal
import subprocess
import urllib.request
from pathlib import Path

VERSION = "26.7.2-0"
INSTALLER = f"Miniforge3-{VERSION}-Linux-x86_64.sh"
URL = f"https://github.com/conda-forge/miniforge/releases/download/{VERSION}/{INSTALLER}"
SHA256 = "281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05"


def environment(base, root, threads=2):
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    for key in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        env.pop(key, None)
    # Do not let a user's pip target/prefix configuration redirect this private install.
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PYTHONPATH"] = str(root / "src")
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    for key, relative in {
        "CONDA_PKGS_DIRS": "cache/conda",
        "PIP_CACHE_DIR": "cache/pip",
        "XDG_CACHE_HOME": "cache/xdg",
        "TORCH_HOME": "cache/torch",
        "MPLCONFIGDIR": "cache/matplotlib",
        "TMPDIR": "cache/tmp",
    }.items():
        folder = base / relative
        folder.mkdir(parents=True, exist_ok=True)
        env[key] = str(folder)
    env["TEMP"] = env["TMP"] = env["TMPDIR"]
    env["CONDA_ENVS_PATH"] = str(base / "envs")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(max(1, int(threads)))
    return env


def stop_process_tree(process):
    if os.name == "posix":

        def send(sig):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

        send(signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # The direct child can exit before a grandchild that ignored SIGTERM.
            send(signal.SIGKILL)
            process.wait()
    else:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_logged(command, env, log):
    with Path(log).open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [str(x) for x in command],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            return process.wait()
        except BaseException:
            stop_process_tree(process)
            raise
        finally:
            process.stdout.close()


def checked(command, env, log):
    code = run_logged(command, env, log)
    if code:
        raise RuntimeError(f"步骤退出码 {code}；详情见 {log}")


def ensure_python(base, root, env, log, conda_path=None):
    prefix = base / "tools/miniforge3"
    conda = Path(conda_path) if conda_path and Path(conda_path).is_file() else prefix / "bin/conda"
    if not conda.is_file():
        if prefix.exists():
            raise ValueError(f"安装目录不完整，请检查或改用新的 runtime：{prefix}")
        download = base / "cache" / INSTALLER
        if not download.exists():
            temporary = download.with_suffix(".partial")
            with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target)
            os.replace(temporary, download)
        digest = hashlib.sha256()
        with download.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != SHA256:
            raise ValueError(f"Miniforge 校验失败，保留文件供排查：{download}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        checked(["bash", download, "-b", "-p", prefix], env, log)
    target = base / "envs/chd_py311"
    python = target / "bin/python"
    if not python.is_file():
        if target.exists():
            raise ValueError(f"Python 环境不完整，请检查或改用新的 runtime：{target}")
        checked([conda, "create", "-y", "-p", target, "python=3.11", "pip"], env, log)
    marker = target / ".chd-installed"
    fingerprint = hashlib.sha256((root / "pyproject.toml").read_bytes() + b"torch2.8.0-cu128-v1").hexdigest()
    if not marker.exists() or marker.read_text().strip() != fingerprint:
        checked(
            [
                python,
                "-m",
                "pip",
                "install",
                "torch==2.8.0",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            env,
            log,
        )
        checked([python, "-m", "pip", "install", str(root)], env, log)
        checked([python, "-m", "pip", "check"], env, log)
        marker.write_text(fingerprint, encoding="utf-8")
    return python
