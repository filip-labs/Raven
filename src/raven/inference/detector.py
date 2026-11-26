"""ObjectDetector adapter over a provider-neutral submit(request) transport boundary."""

import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone

from raven.inference.contracts import (
    create_request,
    request_metadata,
    validate_response,
)


class RemoteDetector:
    def __init__(
        self,
        transport,
        *,
        experiment_id,
        config_sha256,
        preprocessing_sha256,
        response_validator=validate_response,
    ):
        self.transport = transport
        self.response_validator = response_validator
        self.identity = {
            "experiment_id": experiment_id,
            "config_sha256": config_sha256,
            "preprocessing_sha256": preprocessing_sha256,
        }
        self._closed = False
        self._provenance = None
        self._seen = set()

    def initialize(self):
        if self._closed:
            raise RuntimeError("remote detector is closed")
        # No RPC, assets, deployment or model imports during construction/initialize.

    def detect(self, cloud):
        self.initialize()
        self._provenance = None
        started = time.perf_counter()
        request = create_request(
            cloud,
            request_id=uuid.uuid4().hex,
            **self.identity,
            submitted_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        serialized = time.perf_counter() - started
        response, transport = self.transport.submit(deepcopy(request))
        if response.get("request_id") in self._seen:
            raise ValueError("duplicate remote response")
        detections = self.response_validator(response, request)
        if (
            not isinstance(transport, dict)
            or transport.get("provider") != response["worker"]["provider"]
            or transport.get("submission_attempts") != 1
            or not isinstance(transport.get("call_id"), str)
            or not transport["call_id"]
            or len(transport["call_id"]) > 128
        ):
            raise ValueError("remote transport invocation identity unavailable")
        self._seen.add(response["request_id"])
        self._provenance = {
            "backend": "remote",
            "request": request_metadata(request),
            "response": response,
            "transport": transport,
            "client_timings": {
                "clock": "time.perf_counter",
                "serialization_seconds": serialized,
                "total_detector_seconds": time.perf_counter() - started,
                "upload_seconds": None,
                "download_seconds": None,
                "queue_seconds": None,
                "unavailable_reason": "SDK round trip does not expose disjoint transport/queue phases",
            },
        }
        return detections

    @property
    def provenance(self):
        if self._provenance is None:
            raise RuntimeError("no validated remote result")
        return deepcopy(self._provenance)

    def close(self):
        self._closed = True
