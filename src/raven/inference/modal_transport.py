"""Optional authenticated Modal SDK transport. No deployment or implicit resubmission."""

import importlib.metadata
import os
import time

SDK_VERSION = "1.6.4"


class RemoteExecutionError(RuntimeError):
    """Submission failed or outcome is uncertain. Do not automatically resubmit."""


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
        started = time.perf_counter()
        try:
            worker = modal.Cls.from_name(
                self.settings.app,
                self.settings.worker_class,
                environment_name=self.settings.environment,
            )()
            call = worker.infer.spawn(request)
            call_id = call.object_id
            response = call.get(timeout=self.settings.timeout_seconds)
            return response, {
                "provider": "modal",
                "function_call_id": call_id,
                "round_trip_seconds": time.perf_counter() - started,
                "sdk": SDK_VERSION,
                "submission_attempts": 1,
            }
        except Exception as exc:  # noqa: BLE001 -- Never log SDK exception text/credentials.
            cancelled = False
            if call is not None:
                try:
                    call.cancel()
                    cancelled = True
                except Exception:  # noqa: BLE001 -- Best effort; remote timeout remains active.
                    cancelled = False
            raise RemoteExecutionError(
                f"Modal call failed ({type(exc).__name__}); outcome may be uncertain; "
                f"call_id={call.object_id if call is not None else 'unavailable'}, "
                f"cancellation_requested={cancelled}; no resubmission attempted"
            ) from None
