"""Explicit deployment entry point. Import alone creates no cloud resources."""

import os

import modal

from raven.inference.modal_image import build_image
from raven.inference.modal_transport import check_opt_in
from raven.inference.worker import WorkerSession

if modal.is_local():
    check_opt_in()

app = modal.App("raven-pointpillars")


@app.cls(
    image=build_image(),
    gpu="T4",
    cpu=2,
    memory=8192,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=2,
    secrets=(
        [
            modal.Secret.from_name(
                "raven-pointpillars-loading", required_keys=["RAVEN_TRUST_CHECKPOINT"]
            )
        ]
        if os.environ.get("RAVEN_MODAL_TRUSTED_LOADING_SECRET") == "1"
        else []
    ),
    timeout=180,
    startup_timeout=300,
    retries=0,
    volumes={
        "/weights": modal.Volume.from_name("raven-pointpillars-weights", create_if_missing=False)
    },
)
class PointPillarsWorker:
    @modal.enter()
    def initialize(self):
        self.session = WorkerSession()

    @modal.method()
    def infer(self, request):
        return self.session.infer(request)

    @modal.exit()
    def shutdown(self):
        if hasattr(self, "session"):
            self.session.close()
