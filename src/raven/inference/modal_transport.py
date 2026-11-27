"""Optional authenticated Modal SDK transport. No deployment or implicit resubmission."""

import importlib.metadata
import os
import time

SDK_VERSION = "1.6.4"


class RemoteExecutionError(RuntimeError):
    """Submission failed or outcome is uncertain. Do not automatically resubmit."""


def failure_category(exc):
    """Classify by reviewed exception types/codes; never copy arbitrary SDK error text."""
    from raven.inference.worker import WorkerInitializationError

    if isinstance(exc, WorkerInitializationError):
        import re

        match = re.search(r"stage=([a-z_]+); cause=", str(exc))
        stage = match.group(1) if match else "unclassified_startup"
        allowed = {
            "runtime_profile",
            "dependency_import",
            "dependency_source_validation",
            "model_initialization",
            "safe_checkpoint_loading",
            "cuda_ready",
            "unclassified_startup",
        }
        return "worker_startup_failure", stage if stage in allowed else "unclassified_startup"
    name = type(exc).__name__
    if name == "FunctionTimeoutError":
        return "remote_execution_timeout", None
    if isinstance(exc, TimeoutError):
        return "client_result_timeout", None
    if isinstance(exc, PermissionError) or name == "AuthError":
        return "authentication_or_permission", None
    if isinstance(exc, (ConnectionError, OSError)):
        return "transport_failure", None
    if isinstance(exc, ModuleNotFoundError):
        return "remote_dependency_failure", None
    return "remote_or_transport_failure", None


def check_opt_in():
    for name in ("RAVEN_ALLOW_MODAL_GPU", "RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"Modal blocked: explicit {name}=1 required; see Modal guide")


def modal_preflight():
    check_opt_in()
    try:
        version = importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Modal SDK unavailable; install .[modal]") from exc
    if version != SDK_VERSION:
        raise RuntimeError(f"Modal SDK must be {SDK_VERSION}")
    return {
        "source": "modal",
        "sdk": version,
        "authentication": "SDK-managed, checked on call",
        "submission_attempts": 1,
        "cloud_authorization": "explicit_environment_opt_in",
    }


class ModalTransport:
    """One spawn and bounded get. Only Modal handles supported authentication."""

    def __init__(self, settings):
        self.settings = settings

    def submit(self, request):
        check_opt_in()  # Recheck immediately before any remote lookup or submission.
        import modal

        call = None
        call_id = "unavailable"
        phase = "lookup"
        started = time.perf_counter()
        try:
            worker = modal.Cls.from_name(
                self.settings.app,
                self.settings.worker_class,
                environment_name=self.settings.environment,
            )()
            phase = "submission"
            call = worker.infer.spawn(request)
            call_id = call.object_id
            phase = "result_wait"
            response = call.get(timeout=self.settings.timeout_seconds)
            return response, {
                "provider": "modal",
                "function_call_id": call_id,
                "call_id": call_id,
                "round_trip_seconds": time.perf_counter() - started,
                "sdk": SDK_VERSION,
                "submission_attempts": 1,
            }
        except Exception as exc:  # noqa: BLE001 -- Never log SDK exception text/credentials.
            category, stage = failure_category(exc)
            cancelled = False
            definitive = category in {
                "worker_startup_failure",
                "remote_execution_timeout",
                "remote_dependency_failure",
            }
            if call is not None and not definitive:
                try:
                    # Stronger best-effort cleanup; acknowledgement is not terminal-state confirmation.
                    call.cancel(terminate_containers=True)
                    cancelled = True
                except Exception:  # noqa: BLE001 -- Best-effort cleanup, never resubmit.
                    cancelled = False
            outcome = (
                "failed" if definitive else "not_submitted" if phase == "lookup" else "uncertain"
            )
            raise RemoteExecutionError(
                f"Modal call failed ({type(exc).__name__}); category={category}; startup_stage={stage}; "
                f"phase={phase}; outcome={outcome}; call_id={call_id}; "
                f"cancellation_requested={cancelled}; cancellation_confirmed=False; "
                "lookup/spawn are outside result-wait timeout; no resubmission attempted"
            ) from None
