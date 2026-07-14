import re

# Matches things like 412345.67, $412,345.67, -1234
_NUM_RE = re.compile(r'-?\$?\d[\d,]*(?:\.\d+)?')


def _to_float(token):
    return float(token.replace('$', '').replace(',', ''))


def parse_forecast(text, horizon):
    """
    Parse an LLM forecast response into `horizon` floats.

    Returns (values, error): `values` is a list[float] of length `horizon`
    on success, else None; `error` is None on success, else a short reason
    string (the raw response is always kept alongside this in the caller so
    nothing is lost on failure).
    """
    if text is None:
        return None, 'empty response'

    # Preferred path: the model followed the '|'-separated format we asked for.
    if '|' in text:
        parts = [p.strip() for p in text.strip().strip('|').split('|')]
        if len(parts) == horizon:
            values = []
            ok = True
            for p in parts:
                m = _NUM_RE.search(p)
                if not m:
                    ok = False
                    break
                values.append(_to_float(m.group(0)))
            if ok:
                return values, None

    # Fallback: scan the whole response for numeric tokens in order.
    matches = _NUM_RE.findall(text)
    if len(matches) >= horizon:
        # If there are extra matches (e.g. the model appended a trailing
        # caveat sentence with a number in it), take the first `horizon` -
        # the forecast values themselves are expected to come first.
        values = [_to_float(m) for m in matches[:horizon]]
        return values, None
    elif matches:
        return None, f'found {len(matches)}/{horizon} numeric values'
    else:
        return None, 'no numeric values found in response'
