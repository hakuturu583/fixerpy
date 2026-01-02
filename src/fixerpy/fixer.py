from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import docker
from docker.types import DeviceRequest


def _run(
    cmd: List[str], cwd: Optional[Path] = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a shell command with basic logging."""
    print("[fixerpy] $", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def clone_fixer_repo(dest_root: Path) -> Path:
    """Clone nv-tlabs/Fixer into dest_root/Fixer if not exists.

    Returns the path to the cloned repository directory.
    """
    dest_root = Path(dest_root).expanduser().resolve()
    repo_dir = dest_root / "Fixer"
    repo_url = "https://github.com/nv-tlabs/Fixer.git"

    dest_root.mkdir(parents=True, exist_ok=True)
    if (repo_dir / ".git").exists():
        print(f"[fixerpy] Repo already present at {repo_dir}")
        return repo_dir

    _run(["git", "clone", repo_url, str(repo_dir)])
    return repo_dir


def build_docker_cosmos(
    repo_dir: Path,
    tag: str = "fixer-cosmos-env",
    dockerfile: str = "Dockerfile.cosmos",
    platform: str | None = None,
) -> None:
    """Build Fixer Docker image from Dockerfile.cosmos."""
    repo_dir = Path(repo_dir).expanduser().resolve()
    dockerfile_path = repo_dir / dockerfile
    if not dockerfile_path.exists():
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

    # Resolve platform
    if not platform:
        import platform as _p

        m = _p.machine()
        if m in ("x86_64", "amd64"):
            platform = "linux/amd64"
        elif m in ("aarch64", "arm64"):
            platform = "linux/arm64"

    print(
        f"[fixerpy] Building image '{tag}' (platform={platform}) from {dockerfile_path}"
    )
    client = docker.from_env()
    # Use low-level API to pass platform reliably
    api = client.api
    build_stream = api.build(
        path=str(repo_dir),
        dockerfile=str(dockerfile_path.relative_to(repo_dir)),
        tag=tag,
        platform=platform,
        decode=True,
        pull=False,
    )
    for chunk in build_stream:
        if "stream" in chunk:
            line = chunk["stream"].strip()
            if line:
                print(line)
        elif "error" in chunk:
            raise RuntimeError(f"Docker build error: {chunk['error']}")


def download_weights(repo_dir: Path, local_dir: str = "models") -> Path:
    """Download Fixer weights via `hf download nvidia/Fixer --local-dir models` into repo.

    Returns the models directory path.
    """
    repo_dir = Path(repo_dir).expanduser().resolve()
    models_dir = repo_dir / local_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # Use the `hf` CLI as requested.
    _run(["hf", "download", "nvidia/Fixer", "--local-dir", str(models_dir)])
    return models_dir


def run_docker_container(
    repo_dir: Path,
    input_dir: Path,
    output_dir: Path,
    tag: str = "fixer-cosmos-env",
    use_gpus: bool = True,
    extra_args: Optional[List[str]] = None,
    command: Optional[List[str]] = None,
) -> None:
    """Run the Fixer Docker container with mounts for scripts/README and IO dirs.

    - Mounts repo_dir at /work
    - Mounts input_dir at /work/input
    - Mounts output_dir at /work/output
    - Optionally adds --gpus all
    - If command is None, launches bash for interactive usage.
    """
    repo_dir = Path(repo_dir).expanduser().resolve()
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    volumes = {
        str(repo_dir): {"bind": "/work", "mode": "rw"},
        str(input_dir): {"bind": "/work/input", "mode": "rw"},
        str(output_dir): {"bind": "/work/output", "mode": "rw"},
    }

    device_requests = None
    if use_gpus:
        device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])]

    client = docker.from_env()
    cmd = command or ["bash"]
    # extra_args are not directly supported in SDK; ignore or extend as needed.
    print(f"[fixerpy] Running container '{tag}' with command: {' '.join(cmd)}")
    container = client.containers.run(
        image=tag,
        command=cmd,
        volumes=volumes,
        working_dir="/work",
        ipc_mode="host",
        remove=True,
        detach=True,
        device_requests=device_requests,
    )
    # Stream logs until exit
    for line in container.logs(stream=True):
        try:
            print(line.decode().rstrip())
        except Exception:
            print(str(line).rstrip())
    result = container.wait()
    status_code = result.get("StatusCode", 0)
    if status_code != 0:
        raise RuntimeError(f"Container exited with status {status_code}")


def run_inference_container(
    repo_dir: Path,
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    tag: str = "fixer-cosmos-env",
    timestep: int = 250,
    batch_size: int = 1,
    test_speed: bool = False,
    use_gpus: bool = True,
    extra_args: Optional[List[str]] = None,
) -> None:
    """Run the Fixer inference script inside the Docker container per README.

    It mounts the repository at /work, and invokes:
      python /work/src/inference_pretrained_model.py --model /work/models/pretrained/pretrained_fixer.pkl \
        --input /work/input --output /work/output --timestep <timestep>
    """
    repo_dir = Path(repo_dir).expanduser().resolve()
    in_dir = Path(input_dir) if input_dir else repo_dir / "examples"
    out_dir = Path(output_dir) if output_dir else repo_dir.parent / "output"
    in_dir = in_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    volumes = {
        str(repo_dir): {"bind": "/work", "mode": "rw"},
        str(in_dir): {"bind": "/work/input", "mode": "rw"},
        str(out_dir): {"bind": "/work/output", "mode": "rw"},
    }

    device_requests = None
    if use_gpus:
        device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])]

    cmd = [
        "python3",
        "/work/src/inference_pretrained_model.py",
        "--model",
        "/work/models/pretrained/pretrained_fixer.pkl",
        "--input",
        "/work/input",
        "--output",
        "/work/output",
        "--timestep",
        str(timestep),
        "--batch-size",
        str(batch_size),
    ]
    if test_speed:
        cmd.append("--test-speed")

    client = docker.from_env()
    print(f"[fixerpy] Running inference in container '{tag}'")
    container = client.containers.run(
        image=tag,
        command=cmd,
        volumes=volumes,
        working_dir="/work",
        ipc_mode="host",
        remove=True,
        detach=True,
        device_requests=device_requests,
    )
    for line in container.logs(stream=True):
        try:
            print(line.decode().rstrip())
        except Exception:
            print(str(line).rstrip())
    result = container.wait()
    status_code = result.get("StatusCode", 0)
    if status_code != 0:
        raise RuntimeError(f"Container exited with status {status_code}")


def setup_and_run(
    dest_root: Path = Path(".fixer_work"),
    tag: str = "fixer-cosmos-env",
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    use_gpus: bool = True,
    run_cmd: Optional[List[str]] = None,
    extra_docker_args: Optional[List[str]] = None,
) -> None:
    """Full pipeline: clone -> build -> download weights -> run container.

    The container mounts the repo, and input/output directories.
    """
    dest_root = Path(dest_root)
    repo_dir = clone_fixer_repo(dest_root)
    build_docker_cosmos(repo_dir, tag=tag)
    download_weights(repo_dir, local_dir="models")

    in_dir = Path(input_dir) if input_dir else dest_root / "input"
    out_dir = Path(output_dir) if output_dir else dest_root / "output"
    run_docker_container(
        repo_dir=repo_dir,
        input_dir=in_dir,
        output_dir=out_dir,
        tag=tag,
        use_gpus=use_gpus,
        extra_args=extra_docker_args,
        command=run_cmd,
    )


def setup_and_infer(
    dest_root: Path = Path(".fixer_work"),
    tag: str = "fixer-cosmos-env",
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    timestep: int = 250,
    batch_size: int = 1,
    test_speed: bool = False,
    use_gpus: bool = True,
    extra_docker_args: Optional[List[str]] = None,
    platform: str | None = None,
) -> None:
    """Clone -> build -> download weights -> run inference in container per README."""
    dest_root = Path(dest_root)
    repo_dir = clone_fixer_repo(dest_root)
    build_docker_cosmos(repo_dir, tag=tag, platform=platform)
    download_weights(repo_dir, local_dir="models")
    run_inference_container(
        repo_dir=repo_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        tag=tag,
        timestep=timestep,
        batch_size=batch_size,
        test_speed=test_speed,
        use_gpus=use_gpus,
        extra_args=extra_docker_args,
    )


if __name__ == "__main__":
    # Minimal CLI behavior: run the full pipeline with defaults.
    setup_and_run()
