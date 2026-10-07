from pathlib import Path

import yaml


def test_readme_declares_docker_space_on_public_port():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n")
    metadata = yaml.safe_load(readme.split("---", 2)[1])
    assert metadata["sdk"] == "docker"
    assert metadata["app_port"] == 7860
    assert metadata["license"] == "mit"


def test_dockerfile_is_space_compatible_and_health_checked():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "STREAMLIT_SERVER_ADDRESS=0.0.0.0" in dockerfile
    assert "STREAMLIT_SERVER_PORT=7860" in dockerfile
    assert "EXPOSE 7860" in dockerfile
    assert "127.0.0.1:7860/_stcore/health" in dockerfile
    assert "--no-install-project" in dockerfile
    assert "artifacts/operational/residual_v1" in dockerfile


def test_container_does_not_copy_research_datasets():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "COPY data" not in dockerfile
    assert "COPY experiments" not in dockerfile
    assert "COPY tests" not in dockerfile
    assert "COPY artifacts/operational/residual_v1" in dockerfile
