from __future__ import annotations

import tomllib
from pathlib import Path


def test_whisperx_cpu_and_cuda_extras_are_mutually_exclusive() -> None:
    project = Path(__file__).parents[1]
    with (project / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)

    extras = config["project"]["optional-dependencies"]
    assert "whisperx" not in extras
    assert "whisperx-cpu" in extras
    assert "whisperx-cuda" in extras
    assert any(item.startswith("whisperx>=3.8.4") for item in extras["whisperx-cpu"])
    assert any(item.startswith("whisperx>=3.8.4") for item in extras["whisperx-cuda"])

    conflicts = config["tool"]["uv"]["conflicts"]
    assert [
        {"extra": "whisperx-cpu"},
        {"extra": "whisperx-cuda"},
    ] in conflicts


def test_whisperx_extras_select_matching_pytorch_indexes() -> None:
    project = Path(__file__).parents[1]
    with (project / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)

    sources = config["tool"]["uv"]["sources"]
    for package in ("torch", "torchaudio", "torchvision", "torchcodec"):
        assert sources[package] == [
            {"index": "pytorch-cpu", "extra": "whisperx-cpu"},
            {"index": "pytorch-cu128", "extra": "whisperx-cuda"},
        ]

    indexes = {entry["name"]: entry for entry in config["tool"]["uv"]["index"]}
    assert indexes["pytorch-cpu"]["url"].endswith("/whl/cpu")
    assert indexes["pytorch-cu128"]["url"].endswith("/whl/cu128")
    assert indexes["pytorch-cpu"]["explicit"] is True
    assert indexes["pytorch-cu128"]["explicit"] is True


def test_locked_cpu_torch_has_no_nvidia_runtime_dependencies() -> None:
    project = Path(__file__).parents[1]
    with (project / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)

    cpu_torch = next(
        package
        for package in lock["package"]
        if package["name"] == "torch" and package["version"].endswith("+cpu")
    )
    cuda_torch = next(
        package
        for package in lock["package"]
        if package["name"] == "torch" and package["version"].endswith("+cu128")
    )

    assert all(
        not dependency["name"].startswith("nvidia-")
        for dependency in cpu_torch["dependencies"]
    )
    assert any(
        dependency["name"].startswith("nvidia-")
        for dependency in cuda_torch["dependencies"]
    )


def test_cloud_dependencies_are_optional_and_use_official_sdks():
    project = Path(__file__).parents[1]
    with (project / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    extras = config["project"]["optional-dependencies"]
    assert any(item.startswith("dashscope>=") for item in extras["fun-asr"])
    assert any(item.startswith("httpx>=") for item in extras["fun-asr"])
    assert any(item.startswith("azure-cognitiveservices-speech>=1.52") for item in extras["mai"])
    assert any(item.startswith("httpx>=") for item in extras["mai"])
    assert all(not item.startswith(("dashscope", "azure-cognitiveservices-speech", "httpx"))
               for item in config["project"]["dependencies"])
