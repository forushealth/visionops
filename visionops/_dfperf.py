"""Shared runtime-diagnostics lifecycle for training orchestrators."""

from __future__ import annotations

from typing import Optional

from .DfPerfUtils import (
    finish_dfperf_runtime_monitor,
    run_dfperf_preflight,
    start_dfperf_runtime_monitor,
)


class DfPerfLifecycleMixin:
    """Provide the diagnostics lifecycle used by both training pipelines."""

    def _run_pre_training_checks(self) -> None:
        if not self.run_dfperf_preflight:
            self.dfperf_report = {
                "status": "skipped",
                "message": "dfperf preflight disabled",
            }
            return
        self.dfperf_report = run_dfperf_preflight(
            outdir=self.dfperf_outdir,
            strict=self.dfperf_strict,
            sample_gpu=self.dfperf_sample_gpu,
            sample_seconds=self.dfperf_sample_seconds,
            sample_interval=self.dfperf_sample_interval,
        )

    def _start_dfperf_runtime_monitor(self):
        if not self.dfperf_monitor_during_training:
            return None
        return start_dfperf_runtime_monitor(
            outdir=self.dfperf_outdir,
            enable_gpu_sampling=True,
            gpu_interval_s=self.dfperf_sample_interval,
        )

    def _finalize_dfperf_runtime_monitor(
        self,
        session,
        training_ok: bool,
        error_message: Optional[str],
    ) -> None:
        self.dfperf_runtime_report = finish_dfperf_runtime_monitor(
            session,
            backend=self.backend,
            task_type=str(self.task_spec.task_type),
            tag=self.dfperf_tag,
            training_ok=training_ok,
            error_message=error_message,
            generate_report=self.dfperf_generate_report,
            generate_plots=self.dfperf_generate_plots,
        )
