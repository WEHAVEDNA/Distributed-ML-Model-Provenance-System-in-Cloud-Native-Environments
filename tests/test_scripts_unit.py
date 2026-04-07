"""
Unit tests for deployment and helper scripts.

These tests keep the script layer under coverage without needing to execute a
full cluster or Docker workflow.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def _parse_powershell_script(script_path: Path) -> subprocess.CompletedProcess[str]:
    command = (
        "$tokens=$null; $errors=$null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{script_path}', [ref]$tokens, [ref]$errors) > $null; "
        "if ($errors.Count -eq 0) { '[]' } else { $errors | ForEach-Object Message | ConvertTo-Json -Compress }"
    )
    return subprocess.run(
        ["pwsh", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "script_name",
    [
        "build_images.ps1",
        "deploy_k8s.ps1",
        "pipeline.ps1",
        "port_forward.ps1",
        "watch_k8s_status.ps1",
    ],
)
def test_powershell_scripts_parse_cleanly(script_name):
    if shutil.which("pwsh") is None:
        pytest.skip("pwsh is not available")

    result = _parse_powershell_script(SCRIPTS_DIR / script_name)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"{script_name} has parse errors: {result.stdout}"


def test_setup_k8s_bat_runs_deploy_without_opening_new_terminal():
    contents = (SCRIPTS_DIR / "setup_k8s_docker_desktop.bat").read_text(encoding="utf-8").lower()
    assert "\nstart " not in contents
    assert "\nstart\t" not in contents
    assert "deploy_k8s.ps1" in contents
    assert "-buildimages -dockerdesktop" in contents.replace('"', "")
    assert "port_forward.ps1" in contents
    assert "watch_k8s_status.ps1" in contents
    assert "--no-watch" in contents


def test_stop_k8s_port_forward_bat_delegates_to_stop_flag():
    contents = (SCRIPTS_DIR / "stop_k8s_port_forward.bat").read_text(encoding="utf-8").lower()
    assert "port_forward.ps1" in contents
    assert "-stop" in contents


def test_deploy_k8s_script_keeps_local_image_restart_and_port_forward_guidance():
    contents = (SCRIPTS_DIR / "deploy_k8s.ps1").read_text(encoding="utf-8")
    assert "function Reset-Workloads" in contents
    assert "Cleaning existing workloads so startup does not reuse old pods" in contents
    assert "kubectl delete deployment" in contents
    assert "k8s/overlays/docker-desktop" not in contents
    assert "pwsh scripts/port_forward.ps1" in contents


def test_watch_k8s_status_script_streams_pods():
    contents = (SCRIPTS_DIR / "watch_k8s_status.ps1").read_text(encoding="utf-8")
    assert "kubectl get pods" in contents
    assert "--watch" in contents
    assert "port_forward.ps1" in contents


def test_build_images_script_supports_local_cluster_targets():
    contents = (SCRIPTS_DIR / "build_images.ps1").read_text(encoding="utf-8")
    assert "[switch]$DockerDesktop" in contents
    assert "[switch]$Minikube" in contents
    assert "[switch]$Kind" in contents
    assert "minikube image load" in contents
    assert "kind load docker-image" in contents


def test_pipeline_script_health_output_includes_backend_details():
    contents = (SCRIPTS_DIR / "pipeline.ps1").read_text(encoding="utf-8")
    assert 'mode=$($r.deployment_mode)' in contents
    assert '$r.pod_name' in contents
    assert '$r.node_name' in contents
    assert '$SIDECAR_URL/registry?pipeline_id=$PipelineId' in contents


def test_docker_desktop_overlay_sets_local_image_pull_policy():
    manifests = [
        ROOT / "k8s/03-data-ingestion.yaml",
        ROOT / "k8s/04-preprocessing.yaml",
        ROOT / "k8s/05-fine-tuning.yaml",
        ROOT / "k8s/06-atlas-sidecar.yaml",
    ]
    for manifest in manifests:
        contents = manifest.read_text(encoding="utf-8")
        assert "imagePullPolicy: Never" in contents
