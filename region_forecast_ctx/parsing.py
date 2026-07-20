"""
JSON-structured forecast parsing for region_forecast_ctx's predict stage.

Where the baseline (region_forecast/parsing.py, unmodified) asks for and
parses a bare '|'-separated numeric string, this project's predict prompts
(see prompts.py's _structured_output_instruction) ask the LLM to return the
prediction together with a rationale and its claimed information sources as
a single JSON object, so a forecast's reasoning is inspectable, not just its
numbers.

Models don't always comply with a JSON-only instruction (extra prose,
markdown fences, or - in --dry-run - the baseline's own pipe-separated mock
response). To keep the pipeline robust rather than losing runs to strict
parsing, this always falls back to the baseline's tolerant numeric parser
for the `prediction` values specifically, while rationale/information_source
are simply absent when the response wasn't valid JSON.
"""

import json
import re

from region_forecast import parsing as base_parsing

VALID_INFORMATION_SOURCES = ('target_series', 'neighbor_information', 'historical_precedent')

_JSON_OBJECT_RE = re.compile(r'\{.*\}', re.DOTALL)


def _extract_json_object(text):
    m = _JSON_OBJECT_RE.search(text)
    return m.group(0) if m else None


def _clean_prediction(raw, horizon):
    if not isinstance(raw, list) or len(raw) != horizon:
        return None
    try:
        return [float(v) for v in raw]
    except (TypeError, ValueError):
        return None


def _clean_rationale(raw):
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _clean_information_source(raw):
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [s.strip() for s in raw if isinstance(s, str) and s.strip() in VALID_INFORMATION_SOURCES]


def parse_structured_forecast(text, horizon, expect_json=True):
    """
    Returns (pred, rationale, information_source, parse_error):
      pred: list[float] of length `horizon`, or None on failure.
      rationale: str or None (only ever set if the response was valid JSON).
      information_source: list[str], subset of VALID_INFORMATION_SOURCES
        (possibly empty even on success - the model may report using none).
      parse_error: None if `pred` was recovered, else a short reason string
        (mirrors region_forecast.parsing.parse_forecast's error style).

    expect_json: pass cfg.predict_thinking. When False, the prompt never
    asked for JSON (thinking mode is off), so we skip straight to the
    baseline's plain numeric parser rather than risk misreading stray
    braces in a prose response as a JSON object.
    """
    if text is None:
        return None, None, [], 'empty response'

    if not expect_json:
        pred, err = base_parsing.parse_forecast(text, horizon)
        return pred, None, [], err

    blob = _extract_json_object(text)
    obj = None
    if blob is not None:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            obj = None

    if isinstance(obj, dict):
        pred = _clean_prediction(obj.get('prediction'), horizon)
        rationale = _clean_rationale(obj.get('rationale'))
        information_source = _clean_information_source(obj.get('information_source', []))
        if pred is not None:
            return pred, rationale, information_source, None
        # Valid JSON but `prediction` was missing/malformed - still try to
        # recover the numbers from the raw text so a rationale-only slip
        # doesn't sink the whole window, but keep whatever rationale/sources
        # were provided.
        pred, err = base_parsing.parse_forecast(text, horizon)
        return pred, rationale, information_source, err

    # Not JSON at all (model ignored the instruction, or --dry-run's mock
    # pipe-separated response) - fall back to the baseline's tolerant parser.
    pred, err = base_parsing.parse_forecast(text, horizon)
    return pred, None, [], err
