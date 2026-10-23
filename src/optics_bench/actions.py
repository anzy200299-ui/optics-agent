"""Structured tool-call schema. Illegal JSON and out-of-range fields fail closed."""

from __future__ import annotations

import json
from typing import Any


def validate_action(payload: dict[str, Any], protocol: dict[str, Any]) -> tuple[bool, str]:
    spec = protocol["actions"]
    action = payload.get("action")
    if action == "stop":
        return True, "stop"
    if action == "rollback":
        return True, "rollback"
    if action != "run_optimizer":
        return False, f"unknown action: {action}"

    method = payload.get("method")
    parameterization = payload.get("parameterization")
    objective = payload.get("objective")
    iterations = payload.get("iterations")
    fidelity = payload.get("fidelity")
    reason = payload.get("reason_code")

    if method not in spec["methods"]:
        return False, f"illegal method: {method}"
    if parameterization not in spec["parameterizations"]:
        return False, f"illegal parameterization: {parameterization}"
    if objective not in spec["objectives"]:
        return False, f"illegal objective: {objective}"
    if iterations not in spec["iterations"]:
        return False, f"illegal iterations: {iterations}"
    if fidelity not in spec["fidelities"]:
        return False, f"illegal fidelity: {fidelity}"
    if reason not in spec["reason_codes"]:
        return False, f"illegal reason_code: {reason}"
    for banned_method, banned_param in spec["illegal_pairs"]:
        if method == banned_method and parameterization == banned_param:
            return False, f"illegal pair: {method}+{parameterization}"
    return True, "ok"


def parse_action(text: str, protocol: dict[str, Any]) -> tuple[bool, str, dict[str, Any] | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"illegal json: {exc.msg}", None
    if not isinstance(payload, dict):
        return False, "illegal json: not an object", None
    ok, reason = validate_action(payload, protocol)
    return ok, reason, payload if ok else None
