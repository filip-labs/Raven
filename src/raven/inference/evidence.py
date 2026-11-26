"""Validate persisted remote evidence without RPC, model imports or reprocessing points."""

from raven.inference.contracts import digest, duration, validate_response_metadata


def validate_remote_evidence(provenance, record, config):
    return _validate_binding(
        provenance,
        record,
        config.experiment_id,
        config.config_sha256,
        config.to_dict()["preprocessing"],
    )


def _validate_binding(provenance, record, experiment_id, config_sha256, preprocessing):
    if not isinstance(provenance, dict) or provenance.get("backend") != "remote":
        raise ValueError("missing remote execution evidence")
    request = provenance.get("request")
    predictions = validate_response_metadata(provenance.get("response"), request)
    if (
        request["experiment_id"] != experiment_id
        or request["experiment_config_sha256"] != config_sha256
        or request["preprocessing_sha256"] != digest(preprocessing)
        or request["frame_id"] != record["frame_id"]
        or request["input"]["sha256"] != record["processed_xyzi_sha256"]
        or request["input"]["points"] != record["processed_count"]
    ):
        raise ValueError("remote request is not bound to this processed experiment frame")
    transport = provenance.get("transport", {})
    if (
        transport.get("provider") != "modal"
        or transport.get("submission_attempts") != 1
        or not isinstance(transport.get("function_call_id"), str)
        or not transport["function_call_id"].startswith("fc-")
    ):
        raise ValueError("remote invocation identity unavailable")
    if transport.get("call_id", transport["function_call_id"]) != transport["function_call_id"]:
        raise ValueError("remote call identities disagree")
    synthetic = provenance.get("synthetic_test", False)
    if (
        type(synthetic) is not bool
        or type(transport.get("synthetic_test", False)) is not bool
        or synthetic != transport.get("synthetic_test", False)
    ):
        raise ValueError("inconsistent synthetic remote evidence label")
    client = provenance.get("client_timings")
    if not isinstance(client, dict) or client.get("clock") != "time.perf_counter":
        raise ValueError("remote client timing boundary unavailable")
    for key in ("serialization_seconds", "total_detector_seconds"):
        duration(client.get(key))
    for key in ("upload_seconds", "download_seconds", "queue_seconds"):
        if client.get(key) is not None:
            raise ValueError("unsupported inferred transport/queue timing")
    if "round_trip_seconds" in transport:
        duration(transport["round_trip_seconds"])
    return predictions


def validate_archived_remote_run(evidence):
    """Recheck embedded remote contracts without original paths, SDK or model files."""
    config = evidence.configuration
    if (
        evidence.source_type != "experiment"
        or (config or {}).get("model", {}).get("source") != "modal"
    ):
        return
    import hashlib
    import json

    from raven.perception.detection.predictions import detections_from_document

    config_hash = hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    if config.get("schema_version") != 2 or config_hash != evidence.config_sha256:
        raise ValueError("archived Modal configuration identity mismatch")
    kinds = set()
    frames = evidence.payload["results"]["frames"]
    if [row["frame_id"] for row in frames] != list(evidence.frame_ids):
        raise ValueError("archived Modal frame ordering mismatch")
    for row in frames:
        source = row["prediction_source"]
        if (
            source.get("mode") != "remote_adapter"
            or source.get("input_binding") != "executed_on_recorded_processed_cloud"
        ):
            raise ValueError("archived remote source binding missing")
        metadata = source["model_provenance"]
        predictions = _validate_binding(
            metadata, row, config["experiment_id"], config_hash, config["preprocessing"]
        )
        if predictions != detections_from_document(
            row["predictions"], expected_frame_id=row["frame_id"]
        ):
            raise ValueError("archived remote predictions disagree with response")
        kinds.add(
            "synthetic_test"
            if metadata.get("synthetic_test")
            else "declared_remote_modal_inference"
        )
    kind = "synthetic_test" if "synthetic_test" in kinds else "declared_remote_modal_inference"
    if evidence.evidence_kind != kind:
        raise ValueError("archived remote evidence kind mismatch")
