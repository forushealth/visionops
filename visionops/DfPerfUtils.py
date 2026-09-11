"""dfperf-style runtime and preflight helpers for training pipelines."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .GPUUtilisation import GPUSampler, generate_gpu_plots, gpu_snapshot, has_nvidia_smi, summarize_gpu_samples


def run_dfperf_preflight(
    outdir: str = "artifacts/dfperf",
    strict: bool = False,
    sample_gpu: bool = False,
    sample_seconds: float = 5.0,
    sample_interval: float = 1.0,
) -> Dict[str, Any]:
    """Run built-in dfperf preflight before training.

    Outputs in `outdir`:
      - dfperf_preflight.json
      - gpu_samples.csv (optional when sample_gpu=True)
      - gpu_summary.json (optional when sample_gpu=True)
    """
    os.makedirs(outdir, exist_ok=True)
    preflight_json = os.path.join(outdir, "dfperf_preflight.json")

    info: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ok",
        "runner": f"{__name__}.run_dfperf_preflight",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", None),
        "nvidia_smi": "OK" if has_nvidia_smi() else "MISSING",
        "nsys": "OK" if shutil.which("nsys") else "MISSING",
    }

    snap = gpu_snapshot()
    info["gpu_snapshot"] = snap
    info["gpu_count"] = len(snap)

    if sample_gpu:
        from .GPUUtilisation import sample_gpu_utilisation

        samples_csv = os.path.join(outdir, "gpu_samples.csv")
        summary = sample_gpu_utilisation(
            csv_path=samples_csv,
            duration_s=sample_seconds,
            interval_s=sample_interval,
        )
        summary_json = os.path.join(outdir, "gpu_summary.json")
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        info["gpu_samples_csv"] = samples_csv
        info["gpu_summary_json"] = summary_json
        info["gpu_summary"] = summary

    # If strict mode requires nvidia-smi, fail fast
    if strict and info["nvidia_smi"] != "OK":
        info["status"] = "error"
        info["message"] = "nvidia-smi is required in strict mode"
        with open(preflight_json, "w", encoding="utf-8") as file:
            json.dump(info, file, indent=2)
        raise RuntimeError(info["message"])

    with open(preflight_json, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)

    info["preflight_json"] = preflight_json
    return info


def start_dfperf_runtime_monitor(
    outdir: str = "artifacts/dfperf",
    enable_gpu_sampling: bool = True,
    gpu_interval_s: float = 1.0,
) -> Dict[str, Any]:
    """Start runtime monitoring session before training.

    Returns session dict to be passed into `finish_dfperf_runtime_monitor`.
    """
    os.makedirs(outdir, exist_ok=True)

    session: Dict[str, Any] = {
        "outdir": outdir,
        "start_time": time.time(),
        "gpu_samples_csv": os.path.join(outdir, "gpu_samples.csv"),
        "sampler": None,
    }

    if enable_gpu_sampling and has_nvidia_smi():
        sampler = GPUSampler(csv_path=session["gpu_samples_csv"], interval_s=gpu_interval_s, enabled=True)
        sampler.start()
        session["sampler"] = sampler

    return session


def _build_recommendations(metrics: Dict[str, Any]) -> list[Dict[str, Any]]:
    recs: list[Dict[str, Any]] = []

    avg_util = (metrics.get("gpu_monitor") or {}).get("avg_util_gpu")
    wait_score = metrics.get("wait_score")

    if isinstance(avg_util, (int, float)) and avg_util < 35:
        recs.append(
            {
                "title": "Low GPU utilization",
                "why": f"Average GPU utilization is low ({avg_util:.1f}%).",
                "fix": [
                    "Increase dataloader workers and prefetching.",
                    "Enable pinned memory / async data transfer.",
                    "Increase batch size if memory allows.",
                ],
            }
        )

    if isinstance(wait_score, (int, float)) and wait_score > 0.4:
        recs.append(
            {
                "title": "Potential input pipeline bottleneck",
                "why": f"wait_score={wait_score:.2f} suggests data wait is high relative to step time.",
                "fix": [
                    "Cache or shard dataset locally.",
                    "Increase dataloader workers.",
                    "Profile heavy transforms and move augmentations to GPU where possible.",
                ],
            }
        )

    if not recs:
        recs.append(
            {
                "title": "No obvious bottleneck detected",
                "why": "Basic runtime telemetry looks stable.",
                "fix": [
                    "Track trends across runs in report artifacts.",
                    "Compare with larger batch/worker settings for possible throughput gains.",
                ],
            }
        )

    return recs


def _render_report_md(report: Dict[str, Any]) -> str:
    m = report.get("metrics", {})
    gpu = m.get("gpu_monitor") or {}

    lines = []
    lines.append("# dfperf report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- tag: {m.get('tag')}")
    lines.append(f"- backend: {m.get('backend')}")
    lines.append(f"- task_type: {m.get('task_type')}")
    lines.append(f"- duration_s: {m.get('duration_s')}")
    lines.append(f"- avg_gpu_util: {gpu.get('avg_util_gpu')}")
    lines.append(f"- avg_mem_util: {gpu.get('avg_util_mem')}")
    lines.append(f"- avg_power_w: {gpu.get('avg_power_w')}")
    lines.append(f"- avg_mem_used_mb: {gpu.get('avg_mem_used_mb')}")
    lines.append(f"- wait_score: {m.get('wait_score')}")
    lines.append("")
    lines.append("## Recommendations")

    for i, rec in enumerate(report.get("recommendations", []), 1):
        lines.append(f"{i}. **{rec.get('title','')}**")
        lines.append(f"   - why: {rec.get('why','')}")
        for fx in rec.get("fix", []):
            lines.append(f"   - {fx}")
        lines.append("")

    return "\n".join(lines)


def finish_dfperf_runtime_monitor(
    session: Optional[Dict[str, Any]],
    *,
    backend: str,
    task_type: str,
    tag: Optional[str] = None,
    training_ok: bool = True,
    error_message: Optional[str] = None,
    generate_report: bool = True,
    generate_plots: bool = True,
    step_time_p95_ms: Optional[float] = None,
    data_wait_p95_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Stop runtime monitor and generate report artifacts.

    Artifacts (in outdir):
      - gpu_samples.csv (if sampling enabled)
      - gpu_summary.json
      - metrics.json
      - report.json
      - report.md
      - PNG plots (optional)
    """
    if session is None:
        return {"status": "skipped", "message": "No session provided"}

    outdir = session.get("outdir", "artifacts/dfperf")
    os.makedirs(outdir, exist_ok=True)

    sampler = session.get("sampler")
    if sampler is not None:
        sampler.stop()
        sampler.join(timeout=2.0)

    start_time = float(session.get("start_time", time.time()))
    end_time = time.time()
    duration_s = max(0.0, end_time - start_time)

    gpu_csv = session.get("gpu_samples_csv", os.path.join(outdir, "gpu_samples.csv"))
    gpu_summary = summarize_gpu_samples(gpu_csv)

    gpu_summary_json = os.path.join(outdir, "gpu_summary.json")
    with open(gpu_summary_json, "w", encoding="utf-8") as f:
        json.dump(gpu_summary, f, indent=2)

    wait_score = None
    if isinstance(step_time_p95_ms, (int, float)) and step_time_p95_ms > 0 and isinstance(data_wait_p95_ms, (int, float)):
        wait_score = float(data_wait_p95_ms) / float(step_time_p95_ms)

    metrics = {
        "tag": tag or os.path.basename(os.path.abspath(outdir)),
        "backend": backend,
        "task_type": task_type,
        "training_ok": training_ok,
        "error_message": error_message,
        "start_time": start_time,
        "end_time": end_time,
        "duration_s": duration_s,
        "step_time_p95_ms": step_time_p95_ms,
        "data_wait_p95_ms": data_wait_p95_ms,
        "wait_score": wait_score,
        "gpu_monitor": gpu_summary,
    }

    metrics_json = os.path.join(outdir, "metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    artifact_paths = {
        "gpu_samples_csv": gpu_csv if os.path.isfile(gpu_csv) else None,
        "gpu_summary_json": gpu_summary_json,
        "metrics_json": metrics_json,
    }

    if generate_plots and os.path.isfile(gpu_csv):
        plot_paths = generate_gpu_plots(gpu_csv, outdir)
        artifact_paths.update(plot_paths)

    report = {
        "metrics": metrics,
        "recommendations": _build_recommendations(metrics),
        "artifacts": artifact_paths,
    }

    if generate_report:
        report_json = os.path.join(outdir, "report.json")
        report_md = os.path.join(outdir, "report.md")
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        with open(report_md, "w", encoding="utf-8") as f:
            f.write(_render_report_md(report))
        artifact_paths["report_json"] = report_json
        artifact_paths["report_md"] = report_md

    report["status"] = "ok"
    return report
