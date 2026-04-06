#!/usr/bin/env python3
"""
ML Provenance Pipeline demo runner.

Runs ingestion and preprocessing, optionally fine-tunes a model, then prints
pipeline-scoped provenance and status summaries.

Usage:
    python demo.py
    python demo.py --samples 200 --pipeline-id demo-200
    python demo.py --samples 500 --train --pipeline-id demo-500
    python demo.py --samples 500 --train --pipeline-id demo-500 --predict-text "Loved it."

Prerequisites:
    docker compose --progress plain up --build
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


INGEST_URL = "http://localhost:8001"
PREPROC_URL = "http://localhost:8002"
FINETUNE_URL = "http://localhost:8003"
SIDECAR_URL = "http://localhost:8004"

POLL_INTERVAL = 3
HEALTH_TIMEOUT = 180
STAGE_TIMEOUTS = {
    "ingestion": 600,
    "preprocessing": 600,
    "training": 7200,
}
DEFAULT_PREDICT_TEXTS = (
    ("A truly spectacular film. Moved me to tears.", "positive"),
    ("Complete garbage. Poorly directed and acted.", "negative"),
    ("An okay film, nothing special but watchable.", None),
)
STAGE_CHOICES = ("pipeline", "full", "ingest", "preprocess", "train")


def _use_color() -> bool:
    return sys.stdout.isatty() and sys.stderr.isatty()


if _use_color():
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
else:
    BOLD = GREEN = YELLOW = CYAN = RESET = ""


def h1(msg: str):
    print(f"\n{BOLD}{CYAN}{'═' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")


def h2(msg: str):
    print(f"\n{BOLD}── {msg} ──{RESET}")


def ok(msg: str):
    print(f"{GREEN}  [ok]{RESET} {msg}")


def info(msg: str):
    print(f"     {msg}")


def warn(msg: str):
    print(f"{YELLOW}  [warn]{RESET} {msg}")


@dataclass(frozen=True)
class Service:
    name: str
    base_url: str

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"


SERVICES = (
    Service("data-ingestion", INGEST_URL),
    Service("preprocessing", PREPROC_URL),
    Service("fine-tuning", FINETUNE_URL),
    Service("atlas-sidecar", SIDECAR_URL),
)


def _request_json(method: str, url: str, payload: dict | None = None, timeout: int = 10) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:300]}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"Cannot reach {url}: {reason}") from exc

    if not raw:
        return {}
    return json.loads(raw)


def get_json(url: str, timeout: int = 10) -> dict:
    return _request_json("GET", url, timeout=timeout)


def post_json(url: str, payload: dict | None = None, timeout: int = 10) -> dict:
    return _request_json("POST", url, payload=payload, timeout=timeout)


def build_url(base_url: str, path: str, **params: object) -> str:
    query = {key: value for key, value in params.items() if value is not None}
    encoded = urllib.parse.urlencode(query)
    return f"{base_url}{path}" if not encoded else f"{base_url}{path}?{encoded}"


def wait_for_services(timeout_s: int) -> None:
    h2("Service health")
    deadline = time.time() + timeout_s
    pending = {service.name: service for service in SERVICES}

    while pending and time.time() < deadline:
        resolved = []
        for name, service in pending.items():
            try:
                body = get_json(service.health_url, timeout=5)
            except RuntimeError:
                continue

            if body.get("status") != "ok":
                continue

            ok(f"{name:20s} {body.get('status', '?')}")
            if name == "atlas-sidecar":
                info(
                    f"atlas-cli: {body.get('atlas_cli', 'unavailable')}   "
                    f"signing-key: {'present' if body.get('key_exists') else 'missing'}"
                )
            resolved.append(name)

        for name in resolved:
            pending.pop(name, None)

        if pending:
            time.sleep(2)

    if pending:
        missing = ", ".join(sorted(pending))
        raise SystemExit(
            f"\nERROR: services did not become healthy within {timeout_s}s: {missing}\n"
            "Run: docker compose --progress plain up --build"
        )


def wait_for_job(base_url: str, job_id: str, label: str, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            data = get_json(f"{base_url}/jobs/{job_id}", timeout=10)
        except RuntimeError as exc:
            print(f"  {label}: waiting for service ({exc})\r", end="", flush=True)
            time.sleep(POLL_INTERVAL)
            continue

        status = data.get("status", "unknown")
        if status == "completed":
            print(" " * 120, end="\r")
            ok(f"{label} completed")
            return data
        if status == "failed":
            raise SystemExit(f"\nERROR: {label} failed: {data.get('error', 'unknown error')}")
        if status == "cancelled":
            raise SystemExit(f"\n{label} cancelled")

        details = []
        if "epoch" in data:
            details.append(f"epoch={data['epoch']}")
        current_step = data.get("current_step")
        steps_per_epoch = data.get("steps_per_epoch")
        if current_step is not None and steps_per_epoch:
            details.append(f"step={current_step}/{steps_per_epoch}")
        completed_steps = data.get("completed_steps")
        total_steps = data.get("total_steps")
        if completed_steps is not None and total_steps:
            details.append(f"total={completed_steps}/{total_steps}")
        progress_pct = data.get("progress_pct")
        if progress_pct is not None:
            details.append(f"{progress_pct:.1f}%")
        current_loss = data.get("current_loss")
        if current_loss is not None:
            details.append(f"loss={current_loss:.4f}")
        losses = data.get("losses") or data.get("epoch_losses") or []
        if losses and current_loss is None:
            details.append(f"loss={losses[-1]:.4f}")
        suffix = f" ({', '.join(details)})" if details else ""
        print(f"  {label}: {status}{suffix}\r", end="", flush=True)
        time.sleep(POLL_INTERVAL)

    raise SystemExit(f"\nERROR: {label} timed out after {timeout_s}s")


def request_job_cancel(base_url: str, job_id: str, label: str) -> None:
    try:
        data = post_json(f"{base_url}/jobs/{job_id}/cancel", timeout=10)
    except RuntimeError as exc:
        warn(f"Could not request remote cancellation for {label}: {exc}")
        return

    print(" " * 120, end="\r")
    warn(
        f"{label} interruption requested. "
        f"Remote job {job_id} status={data.get('status', 'unknown')}"
    )


def wait_for_job_with_cancel(base_url: str, job_id: str, label: str, timeout_s: int) -> dict:
    try:
        return wait_for_job(base_url, job_id, label, timeout_s)
    except KeyboardInterrupt:
        request_job_cancel(base_url, job_id, label)
        raise SystemExit(f"\n{label} interrupted locally. Remote cancellation requested.")


def print_stage_result(result: dict, sha_field: str = "sha256") -> None:
    for key in ("pipeline_id", "num_samples", "max_length", "epochs", "device", "s3_uri"):
        value = result.get(key)
        if value is not None:
            info(f"{key:10s}: {value}")
    digest = result.get(sha_field)
    if digest:
        info(f"{sha_field:10s}: {digest[:16]}…")
    epoch_losses = result.get("epoch_losses")
    if epoch_losses:
        info(f"epoch_losses: {epoch_losses}")
    manifest_id = result.get("manifest_id")
    if manifest_id:
        ok(f"Atlas manifest: {manifest_id}")
    else:
        warn("No Atlas manifest recorded for this stage")


def run_ingestion(split: str, samples: int, pipeline_id: str) -> dict:
    h2("Stage 1 – Data Ingestion")
    info(f"split={split}   samples={samples}   pipeline_id={pipeline_id}")
    response = post_json(
        build_url(
            INGEST_URL,
            "/ingest",
            split=split,
            num_samples=samples,
            pipeline_id=pipeline_id,
        )
    )
    info(f"job_id    : {response['job_id']}")
    result = wait_for_job_with_cancel(
        INGEST_URL,
        response["job_id"],
        "Ingestion",
        STAGE_TIMEOUTS["ingestion"],
    )
    print_stage_result(result)
    return result


def run_preprocessing(split: str, pipeline_id: str) -> dict:
    h2("Stage 2 – Preprocessing")
    info(f"split={split}   pipeline_id={pipeline_id}")
    response = post_json(
        build_url(PREPROC_URL, "/preprocess", split=split, pipeline_id=pipeline_id)
    )
    info(f"job_id    : {response['job_id']}")
    result = wait_for_job_with_cancel(
        PREPROC_URL,
        response["job_id"],
        "Preprocessing",
        STAGE_TIMEOUTS["preprocessing"],
    )
    print_stage_result(result)
    return result


def run_training(split: str, pipeline_id: str, epochs: int) -> dict:
    h2("Stage 3 – Fine-Tuning")
    warn(f"CPU training is slow. Demo training defaults to {epochs} epoch(s).")
    response = post_json(
        build_url(FINETUNE_URL, "/train", split=split, pipeline_id=pipeline_id, epochs=epochs)
    )
    info(f"job_id    : {response['job_id']}")
    info(f"device    : {response.get('device')}")
    info(f"epochs    : {response.get('epochs')}")
    result = wait_for_job_with_cancel(
        FINETUNE_URL,
        response["job_id"],
        "Fine-Tuning",
        STAGE_TIMEOUTS["training"],
    )
    print_stage_result(result)
    return result


def show_lineage(pipeline_id: str) -> None:
    h2("Provenance chain")
    try:
        lineage = get_json(build_url(SIDECAR_URL, "/lineage", pipeline_id=pipeline_id), timeout=10)
    except RuntimeError as exc:
        warn(f"Could not fetch lineage: {exc}")
        return

    chain = lineage.get("chain", [])
    if not chain:
        warn("No manifests recorded yet for this pipeline")
        return

    for entry in chain:
        print(
            f"  {BOLD}{entry.get('stage', '?'):20s}{RESET} "
            f"{entry.get('type', '?'):10s} "
            f"manifest={entry.get('manifest_id', 'none')}"
        )
        info(f"  └─ {entry.get('artifact_uri', '')}")

    completed = ", ".join(lineage.get("stages_complete", [])) or "none"
    info(f"stages_complete: {completed}")
    if lineage.get("chain_complete"):
        ok("Full provenance chain complete")
    else:
        warn("Chain incomplete for this pipeline")


def show_pipeline_status(pipeline_id: str) -> None:
    h2("Pipeline status")
    try:
        status = get_json(build_url(SIDECAR_URL, "/pipeline/status", pipeline_id=pipeline_id), timeout=10)
    except RuntimeError as exc:
        warn(f"Could not fetch pipeline status: {exc}")
        return

    for stage, stage_info in status.get("stages", {}).items():
        marker = "ok" if stage_info.get("done") else ".."
        print(f"  [{marker}] {stage:25s} {stage_info.get('artifact_count', 0)} artifact(s)")


def prove_provenance(pipeline_id: str) -> None:
    h2("Provenance proof")

    try:
        lineage = get_json(build_url(SIDECAR_URL, "/lineage", pipeline_id=pipeline_id), timeout=10)
        status = get_json(build_url(SIDECAR_URL, "/pipeline/status", pipeline_id=pipeline_id), timeout=10)
        registry = get_json(build_url(SIDECAR_URL, "/registry", pipeline_id=pipeline_id), timeout=10)
    except RuntimeError as exc:
        warn(f"Could not fetch provenance proof data: {exc}")
        return

    chain = lineage.get("chain", [])
    if not chain:
        warn("No lineage entries found, so provenance proof could not be established")
        return

    if registry:
        ok(f"Registry contains {len(registry)} artifact(s) for pipeline {pipeline_id}")
    else:
        warn("Registry is empty for this pipeline")

    manifest_ids = [entry.get("manifest_id") for entry in chain if entry.get("manifest_id")]
    if len(manifest_ids) == len(chain):
        ok("Every lineage entry has a manifest ID")
    else:
        warn("One or more lineage entries are missing a manifest ID")

    stages = status.get("stages", {})
    done_stages = [name for name, stage_info in stages.items() if stage_info.get("done")]
    if done_stages:
        ok(f"Pipeline status marks these stages complete: {', '.join(done_stages)}")
    else:
        warn("Pipeline status did not mark any stages complete")

    service_provenance_urls = (
        ("data-ingestion", build_url(INGEST_URL, "/provenance", pipeline_id=pipeline_id)),
        ("preprocessing", build_url(PREPROC_URL, "/provenance", pipeline_id=pipeline_id)),
        ("fine-tuning", build_url(FINETUNE_URL, "/provenance", pipeline_id=pipeline_id)),
    )
    for service_name, url in service_provenance_urls:
        try:
            body = get_json(url, timeout=10)
        except RuntimeError as exc:
            warn(f"{service_name} provenance check failed: {exc}")
            continue

        manifest_id = body.get("manifest_id")
        if manifest_id:
            ok(f"{service_name} reports manifest_id={manifest_id}")
        else:
            info(f"{service_name:20s} manifest_id=None")

    manifest_id = manifest_ids[0]
    encoded_manifest_id = urllib.parse.quote(manifest_id, safe="")
    try:
        export_body = get_json(f"{SIDECAR_URL}/export/{encoded_manifest_id}", timeout=30)
        verify_body = get_json(f"{SIDECAR_URL}/verify/{encoded_manifest_id}", timeout=30)
    except RuntimeError as exc:
        warn(f"Could not export/verify manifest {manifest_id}: {exc}")
        return

    exported_root = export_body.get("root_id") or export_body.get("id") or manifest_id
    ok(f"Export succeeded for manifest {exported_root}")
    if "valid" in verify_body:
        validity = "valid" if verify_body.get("valid") else "invalid"
        ok(f"Verify returned valid={verify_body.get('valid')} ({validity})")
    else:
        warn(f"Verify response did not include a 'valid' field: {verify_body}")

    if lineage.get("chain_complete"):
        ok("Proof complete: full provenance chain was collected and linked")
    else:
        warn("Proof partial: provenance exists, but the full chain is not complete for this pipeline")


def run_inference_smoke(pipeline_id: str, predict_text: str | None) -> None:
    h2("Inference smoke test")
    samples = ((predict_text, None),) if predict_text else DEFAULT_PREDICT_TEXTS

    for text, expected in samples:
        try:
            result = post_json(
                f"{FINETUNE_URL}/predict",
                {"text": text, "pipeline_id": pipeline_id},
                timeout=30,
            )
        except RuntimeError as exc:
            warn(f"Predict failed: {exc}")
            return

        label = result.get("label", "?")
        confidence = result.get("confidence", 0.0)
        passed = expected is None or label == expected
        marker = "ok" if passed else "!!"
        expected_suffix = "" if expected is None else f" expected={expected}"
        print(f"  [{marker}] {label:8s} {confidence:.3f}{expected_suffix}  {text[:72]}")


def print_summary(pipeline_id: str) -> None:
    h2("Explore further")
    print(f"  Sidecar registry : {build_url(SIDECAR_URL, '/registry', pipeline_id=pipeline_id)}")
    print(f"  Pipeline status  : {build_url(SIDECAR_URL, '/pipeline/status', pipeline_id=pipeline_id)}")
    print(f"  Provenance chain : {build_url(SIDECAR_URL, '/lineage', pipeline_id=pipeline_id)}")
    print("  Interactive docs : http://localhost:8001/docs  (and 8002, 8003, 8004)")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML Provenance Pipeline demo")
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="IMDB samples to ingest (default: 100)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--pipeline-id",
        default="default",
        help="Pipeline namespace to run under (default: default)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Include fine-tuning and inference smoke tests (legacy alias for --stage full)",
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        default=None,
        help=(
            "Run only one stage or pipeline slice: "
            "pipeline=ingest+preprocess, full=ingest+preprocess+train, "
            "ingest, preprocess, train"
        ),
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
        help="Epochs to use for demo training (default: 1)",
    )
    parser.add_argument(
        "--predict-text",
        default=None,
        help="Custom text to score during the inference smoke test",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=HEALTH_TIMEOUT,
        help=f"Seconds to wait for services to become healthy (default: {HEALTH_TIMEOUT})",
    )
    args = parser.parse_args()
    if args.stage and args.train:
        parser.error("use either --train or --stage, not both")
    args.stage = args.stage or ("full" if args.train else "pipeline")
    return args


def main() -> None:
    args = parse_args()

    h1("ML Provenance Pipeline Demo")
    info(
        f"samples={args.samples}   split={args.split}   "
        f"pipeline_id={args.pipeline_id}   stage={args.stage}"
    )

    wait_for_services(args.health_timeout)
    if args.stage == "ingest":
        run_ingestion(args.split, args.samples, args.pipeline_id)
    elif args.stage == "preprocess":
        warn("Preprocess-only mode expects raw data for this pipeline and split to already exist.")
        run_preprocessing(args.split, args.pipeline_id)
    elif args.stage == "train":
        warn("Train-only mode expects preprocessed data for this pipeline and split to already exist.")
        run_training(args.split, args.pipeline_id, args.train_epochs)
        run_inference_smoke(args.pipeline_id, args.predict_text)
    else:
        run_ingestion(args.split, args.samples, args.pipeline_id)
        run_preprocessing(args.split, args.pipeline_id)
        if args.stage == "full":
            run_training(args.split, args.pipeline_id, args.train_epochs)
            run_inference_smoke(args.pipeline_id, args.predict_text)
        else:
            warn("Skipping fine-tuning. Use --stage full or --stage train to include model training and inference.")

    show_lineage(args.pipeline_id)
    show_pipeline_status(args.pipeline_id)
    prove_provenance(args.pipeline_id)
    print_summary(args.pipeline_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("\nCancelled by user.")
