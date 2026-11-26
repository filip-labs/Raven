"""Local CUDA only. No GPU/remote framework imports occur in CPU benchmark mode."""

import platform
from types import SimpleNamespace


class LocalCudaBackend:
    def __init__(self, config):
        if config.model.source != "inference":
            raise RuntimeError(
                "saved predictions are not fresh inference; select local inference source"
            )
        if platform.system() == "Darwin":
            raise RuntimeError("local CUDA inference unavailable on macOS; no CPU/remote fallback")
        from raven.experiments.runner import _inference_preflight

        self.config = config
        self.evidence = _inference_preflight(config)
        import torch

        self.torch = torch
        self.detector = None

    def synchronize(self):
        self.torch.cuda.synchronize(self.evidence["device_index"])

    def initialize(self):
        from raven.experiments.artifacts import file_sha256
        from raven.experiments.runner import _build_detector

        if file_sha256(self.config.model.config) != self.evidence["config_sha256"]:
            raise ValueError("model configuration changed since preflight")
        self.detector = _build_detector(self.config, SimpleNamespace(model_evidence=self.evidence))
        self.detector.initialize()

    def detect(self, cloud):
        return self.detector.detect(cloud)

    @property
    def provenance(self):
        return self.detector.provenance

    def close(self):
        if self.detector is not None:
            self.detector.close()
