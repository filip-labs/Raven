"""Validate persisted remote evidence without RPC, model imports or reprocessing points."""

from raven.inference.contracts import digest, validate_response_metadata


def validate_remote_evidence(provenance, record, config):
    if not isinstance(provenance, dict) or provenance.get("backend") != "remote":
        raise ValueError("missing remote execution evidence")
    request = provenance.get("request")
    predictions = validate_response_metadata(provenance.get("response"), request)
    if (
        request["experiment_id"] != config.experiment_id
        or request["experiment_config_sha256"] != config.config_sha256
        or request["preprocessing_sha256"] != digest(config.to_dict()["preprocessing"])
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
    return predictions
