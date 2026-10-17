"""Fixed-origin provider discovery and explicit, low-budget connection checks.

Only sanitized model metadata and locally written status messages leave this
module. Checks run only on an operator request; normal dashboard polling never
generates model completions or paid searches.
"""

import re
import time

import httpx

from .clients import _HTTP, MAX_WEB_BYTES, ServiceError, WebResearch, _json, _key

CRUSOE_BASE_URL = "https://api.inference.crusoecloud.com/v1"
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:@+-]{0,199}\Z")
_PARAMETER = re.compile(r"[a-z][a-z_]{0,63}\Z")
_CHECK_TIMEOUT = httpx.Timeout(8, connect=4, write=4, pool=4).as_dict()
_TOOL_NAME = "connection_check"


def _provider_key(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ServiceError("invalid_configuration")
    return _key(value)


def _model_id(value, key=None):
    return (
        isinstance(value, str)
        and bool(_MODEL_ID.fullmatch(value))
        and (not key or key not in value)
    )


def provider_models(key, *, client=None, _deadline=None):
    """Return Crusoe model IDs and declared capabilities, with no raw metadata.

    ``client`` is a test seam; production creates a proxy-free, redirect-denying
    client and always sends credentials to the official inference endpoint.
    """
    key = _provider_key(key)
    with _HTTP(client) as connection:
        result = connection.request(
            "GET",
            CRUSOE_BASE_URL + "/models",
            key=key,
            maximum=1024 * 1024,
            extensions={"timeout": _CHECK_TIMEOUT},
            deadline=_deadline,
        )
    items = result.get("data")
    if not isinstance(items, list) or not items or len(items) > 512:
        raise ServiceError("invalid_response")
    models, seen = [], set()
    for item in items:
        if not isinstance(item, dict) or not _model_id(item.get("id"), key):
            raise ServiceError("invalid_response")
        model_id = item["id"]
        if model_id in seen:
            raise ServiceError("invalid_response")
        seen.add(model_id)
        parameters = item.get("supported_parameters", [])
        if (
            not isinstance(parameters, list)
            or len(parameters) > 100
            or any(
                not isinstance(value, str) or not _PARAMETER.fullmatch(value) or key in value
                for value in parameters
            )
        ):
            raise ServiceError("invalid_response")
        models.append({"id": model_id, "supported_parameters": sorted(set(parameters))})
    return sorted(models, key=lambda model: model["id"])


def _assistant(result):
    try:
        choices = result["choices"]
        if not isinstance(choices, list) or not choices or len(choices) > 10:
            raise ValueError
        message = choices[0]["message"]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError
        return message
    except (KeyError, IndexError, TypeError, ValueError):
        raise ServiceError("invalid_model_response") from None


def _confirmed_json(value):
    if not isinstance(value, str):
        return False
    decoded = _json(value)
    return isinstance(decoded, dict) and set(decoded) == {"ok"} and decoded["ok"] is True


def _check_crusoe(key, model, client):
    if not _model_id(model, key):
        return {"ok": False, "message": "Choose a Crusoe model before testing the connection."}
    # One deadline covers catalog discovery and both paid probes. A blocked
    # read can add at most the eight-second socket timeout to this 25s budget.
    deadline = time.monotonic() + 25
    models = provider_models(key, client=client, _deadline=deadline)
    selected = next((entry for entry in models if entry["id"] == model), None)
    if selected is None:
        return {"ok": False, "message": "The selected model is unavailable for this API key."}
    parameters = selected["supported_parameters"]
    if parameters and not {"tools", "response_format"}.issubset(parameters):
        return {
            "ok": False,
            "message": "Choose a model that supports tool calling and JSON response format.",
        }
    with _HTTP(client) as connection:
        tool = connection.request(
            "POST",
            CRUSOE_BASE_URL + "/chat/completions",
            key=key,
            maximum=MAX_WEB_BYTES,
            extensions={"timeout": _CHECK_TIMEOUT},
            deadline=deadline,
            body={
                "model": model,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": "Call connection_check with ok=true. Give no explanation.",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": _TOOL_NAME,
                            "description": "Confirm the connection works.",
                            "parameters": {
                                "type": "object",
                                "properties": {"ok": {"type": "boolean"}},
                                "required": ["ok"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )
        calls = _assistant(tool).get("tool_calls")
        valid = False
        if isinstance(calls, list) and 0 < len(calls) <= 20:
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    continue
                function = call.get("function")
                if not isinstance(function, dict) or function.get("name") != _TOOL_NAME:
                    continue
                try:
                    valid = (
                        isinstance(call.get("id"), str)
                        and 0 < len(call["id"]) <= 200
                        and _confirmed_json(function.get("arguments"))
                    )
                except (TypeError, ValueError, RecursionError):
                    valid = False
                if valid:
                    break
        if not valid:
            return {
                "ok": False,
                "message": "The model did not pass the tool calling check. Try another model.",
            }
        structured = connection.request(
            "POST",
            CRUSOE_BASE_URL + "/chat/completions",
            key=key,
            maximum=MAX_WEB_BYTES,
            extensions={"timeout": _CHECK_TIMEOUT},
            deadline=deadline,
            body={
                "model": model,
                "max_tokens": 256,
                "messages": [
                    {"role": "system", "content": "Respond only with a JSON object."},
                    {"role": "user", "content": 'Return exactly {"ok":true} as JSON.'},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        try:
            valid = _confirmed_json(_assistant(structured).get("content"))
        except (TypeError, ValueError, RecursionError):
            valid = False
        if not valid:
            return {
                "ok": False,
                "message": "The model did not pass the JSON response check. Try another model.",
            }
    return {"ok": True, "message": "Connected. Model tool calling and JSON responses passed."}


def check_provider(provider, key, model=None, *, client=None):
    """Check stored credentials without returning provider bodies or exceptions."""
    try:
        key = _provider_key(key)
        if provider == "crusoe":
            return _check_crusoe(key, model, client)
        if provider == "you":
            with WebResearch(search_key=key, provider="you", client=client) as web:
                web.search(
                    "You.com search API documentation",
                    timeout=_CHECK_TIMEOUT,
                    deadline=time.monotonic() + 25,
                )
            return {"ok": True, "message": "Connected. You.com search returned a valid response."}
        return {"ok": False, "message": "This provider is not supported."}
    except ServiceError as error:
        messages = {
            "invalid_configuration": "Add a valid API key before testing the connection.",
            "timeout": "The provider timed out. Try the connection test again.",
            "upstream_rejected": "The provider rejected the request. Check the key and account.",
            "unavailable": "The provider could not be reached. Try again later.",
        }
        return {
            "ok": False,
            "message": messages.get(error.code, "The provider returned an unsupported response."),
        }
