"""GPU sampling helpers based on `nvidia-smi` output."""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Dict, List, Optional


def has_nvidia_smi() -> bool:
    """Return ``True`` when ``nvidia-smi`` is available on PATH."""
    return shutil.which("nvidia-smi") is not None


def _safe_float(value: Optional[str]) -> Optional[float]:
    """Convert a string-like value to float and return ``None`` on failure."""
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _run_nvidia_smi_query() -> List[str]:
    """Run the configured `nvidia-smi` query and return raw CSV rows."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []

    rows = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
    return rows


def gpu_snapshot() -> List[Dict[str, Optional[float]]]:
    """Return a one-shot GPU utilization snapshot from nvidia-smi."""
    if not has_nvidia_smi():
        return []

    rows = _run_nvidia_smi_query()
    result: List[Dict[str, Optional[float]]] = []

    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 7:
            continue

        item = {
            "gpu_index": int(parts[0]) if parts[0].isdigit() else None,
            "name": parts[1],
            "util_gpu": _safe_float(parts[2]),
            "util_mem": _safe_float(parts[3]),
            "mem_used_mb": _safe_float(parts[4]),
            "mem_total_mb": _safe_float(parts[5]),
            "power_w": _safe_float(parts[6]),
        }
        result.append(item)

    return result


class GPUSampler(threading.Thread):
    """Background GPU sampler writing `gpu_samples.csv` during training."""

    def __init__(self, csv_path: str, interval_s: float = 1.0, enabled: bool = True) -> None:
        """Initialize a daemon sampler thread.

        Args:
            csv_path: Output CSV path for samples.
            interval_s: Sampling interval in seconds.
            enabled: Whether sampling should run when started.
        """
        super().__init__(daemon=True)
        self.csv_path = csv_path
        self.interval_s = float(interval_s)
        self.enabled = enabled and has_nvidia_smi()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Request the sampler loop to stop."""
        self._stop_event.set()

    def _write_rows(self, writer: csv.DictWriter, rows: List[Dict[str, Optional[float]]]) -> None:
        now = time.time()
        for row in rows:
            writer.writerow({"timestamp": now, **row})

    def run(self) -> None:
        """Continuously write GPU snapshots until stopped."""
        if not self.enabled:
            return

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "timestamp",
                "gpu_index",
                "name",
                "util_gpu",
                "util_mem",
                "mem_used_mb",
                "mem_total_mb",
                "power_w",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while not self._stop_event.is_set():
                snap = gpu_snapshot()
                self._write_rows(writer, snap)
                f.flush()
                self._stop_event.wait(self.interval_s)


def read_gpu_samples(csv_path: str) -> List[Dict[str, Optional[float]]]:
    """Read GPU sample rows from a CSV path.

    Args:
        csv_path: Path produced by :class:`GPUSampler`.

    Returns:
        Parsed rows with normalized numeric fields.
    """
    rows: List[Dict[str, Optional[float]]] = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "timestamp": _safe_float(row.get("timestamp")),
                        "gpu_index": int(row["gpu_index"]) if row.get("gpu_index") not in {None, ""} else None,
                        "name": row.get("name"),
                        "util_gpu": _safe_float(row.get("util_gpu")),
                        "util_mem": _safe_float(row.get("util_mem")),
                        "mem_used_mb": _safe_float(row.get("mem_used_mb")),
                        "mem_total_mb": _safe_float(row.get("mem_total_mb")),
                        "power_w": _safe_float(row.get("power_w")),
                    }
                )
    except FileNotFoundError:
        return []
    return rows


def summarize_gpu_samples(csv_path: str) -> Dict[str, Optional[float]]:
    """Compute aggregate statistics from sampled GPU CSV rows."""
    rows = read_gpu_samples(csv_path)
    if not rows:
        return {
            "avg_util_gpu": None,
            "avg_util_mem": None,
            "avg_mem_used_mb": None,
            "avg_power_w": None,
            "samples": 0,
        }

    def _avg(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    return {
        "avg_util_gpu": _avg("util_gpu"),
        "avg_util_mem": _avg("util_mem"),
        "avg_mem_used_mb": _avg("mem_used_mb"),
        "avg_power_w": _avg("power_w"),
        "samples": len(rows),
    }


def sample_gpu_utilisation(
    csv_path: Optional[str] = None,
    duration_s: float = 5.0,
    interval_s: float = 1.0,
    *,
    seconds: Optional[float] = None,
    interval: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """One-shot helper to sample for fixed duration and return summary.

    Supports aliases used in notebooks:
    - ``seconds`` -> ``duration_s``
    - ``interval`` -> ``interval_s``
    """
    if seconds is not None:
        duration_s = float(seconds)
    if interval is not None:
        interval_s = float(interval)

    if duration_s <= 0 or interval_s <= 0:
        raise ValueError("duration_s and interval_s must be positive")

    cleanup_path = False
    if not csv_path:
        fd, temp_path = tempfile.mkstemp(prefix="visionops_gpu_samples_", suffix=".csv")
        try:
            csv_path = temp_path
            cleanup_path = True
        finally:
            try:
                import os

                os.close(fd)
            except Exception:
                pass

    sampler = GPUSampler(csv_path=csv_path, interval_s=interval_s, enabled=True)
    sampler.start()
    time.sleep(duration_s)
    sampler.stop()
    sampler.join(timeout=interval_s + 1.0)

    summary = summarize_gpu_samples(csv_path)
    if cleanup_path:
        try:
            import os

            if os.path.exists(csv_path):
                os.remove(csv_path)
        except Exception:
            pass
    return summary


def generate_gpu_plots(csv_path: str, outdir: str) -> Dict[str, str]:
    """Generate PNG plots from gpu_samples.csv if matplotlib is available."""
    rows = read_gpu_samples(csv_path)
    if not rows:
        return {}

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    os_paths = {}
    by_idx: Dict[int, List[Dict[str, Optional[float]]]] = {}
    for row in rows:
        idx = row.get("gpu_index")
        if idx is None:
            continue
        by_idx.setdefault(int(idx), []).append(row)

    if not by_idx:
        return {}

    import os

    os.makedirs(outdir, exist_ok=True)

    for idx, samples in by_idx.items():
        t0 = samples[0]["timestamp"] or 0.0
        xs = [(s["timestamp"] or 0.0) - t0 for s in samples]

        # Utilization plot
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(xs, [s["util_gpu"] or 0.0 for s in samples], label="GPU util %")
        ax.plot(xs, [s["util_mem"] or 0.0 for s in samples], label="Mem util %")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("utilization (%)")
        ax.set_title(f"GPU {idx} utilization")
        ax.legend()
        ax.grid(alpha=0.3)
        util_path = os.path.join(outdir, f"gpu{idx}_util.png")
        fig.tight_layout()
        fig.savefig(util_path)
        plt.close(fig)
        os_paths[f"gpu{idx}_util_plot"] = util_path

        # Memory / power plot
        fig, ax1 = plt.subplots(figsize=(8, 3.5))
        ax1.plot(xs, [s["mem_used_mb"] or 0.0 for s in samples], color="tab:blue", label="Mem used MB")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("memory (MB)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(xs, [s["power_w"] or 0.0 for s in samples], color="tab:red", label="Power W")
        ax2.set_ylabel("power (W)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        fig.suptitle(f"GPU {idx} memory and power")
        mem_path = os.path.join(outdir, f"gpu{idx}_mem_power.png")
        fig.tight_layout()
        fig.savefig(mem_path)
        plt.close(fig)
        os_paths[f"gpu{idx}_mem_power_plot"] = mem_path

    return os_paths
