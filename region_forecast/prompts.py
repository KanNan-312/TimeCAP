"""
Prompt templates for region-level home price forecasting, following the same
P1 (contextualize) / P2 (predict from time series) / P3 (predict from text,
"TimeCP") / P4 (predict from text + in-context examples, "TimeCAP") structure
used by the finance/healthcare/weather domains in this repo, adapted from
single-step classification to multi-step (horizon-month) numeric forecasting.

Region identity: zipcode only. city/metro are kept in region_meta.json for
post-hoc grouping/reporting but are deliberately never passed into a prompt -
they're metadata correlated with price level and would leak information
about the region's market segment beyond what the lookback window itself
provides. Target: price. Time aspect: monthly date.
"""

DISPLAY_NAMES = {
    'price': 'Housing Value Index',
    'median_sale_price': 'Median Sale Price (USD)',
    'median_list_price': 'Median List Price (USD)',
    'homes_sold': 'Homes Sold',
    'pending_sales': 'Pending Sales',
    'new_listings': 'New Listings',
    'inventory': 'Inventory (Active Listings)',
    'median_dom': 'Median Days on Market',
    'avg_sale_to_list': 'Average Sale-to-List Ratio',
    'total population': 'Total Population',
}


def display_name(col):
    return DISPLAY_NAMES.get(col, DISPLAY_NAMES.get(col.lower(), col.replace('_', ' ').title()))


def target_label(cfg):
    return display_name(cfg.target_column).lower()


def format_series(values):
    vals = [float(v) for v in values]
    scale = max((abs(v) for v in vals), default=0.0)
    fmt = '{:.4f}' if scale < 10 else '{:.2f}'
    return '|'.join(fmt.format(v) for v in vals)


def indicator_block(cfg, window, target_only=False):
    lines = [f"- {display_name(cfg.target_column)}: {format_series(window['target'])}"]
    if not target_only:
        for col, vals in window['indicators'].items():
            lines.append(f"- {display_name(col)}: {format_series(vals)}")
    return '\n'.join(lines)


def region_label(region_id):
    return f"ZIP code {region_id}"


def forecast_instruction(cfg, forecast_dates):
    return (
        f"forecast the {target_label(cfg)} for each of the next {cfg.horizon} months, "
        f"from {forecast_dates[0]} to {forecast_dates[-1]}. "
        f"Respond with exactly {cfg.horizon} numeric values separated by '|' tokens, in chronological "
        f"order, with no other text, labels, units, or currency symbols "
        f"(e.g. 412345.67|415012.10|...). Do not provide any other details or explanation."
    )


# ---------------------------------------------------------------------------
# P1. Contextualization of the time series
# ---------------------------------------------------------------------------

def contextualize_prompt(cfg, region_id, window):
    label = region_label(region_id)
    system_prompt = (
        "Your job is to act as a professional regional real-estate market analyst. You will write a "
        "high-quality report that is informative and helps in understanding the current regional "
        "housing market situation."
    )
    user_prompt = (
        f"Your task is to analyze key housing market indicators in {label} over the last "
        f"{cfg.lookback} months (from {window['dates'][0]} to {window['dates'][-1]}). "
        f"Review the time-series data provided for the last {cfg.lookback} months. Each time-series "
        f"consists of monthly values separated by a '|' token for the following indicators:\n"
        f"{indicator_block(cfg, window)}\n\n"
        f"Based on this time-series data, write a concise report that provides insights crucial for "
        f"understanding the current regional housing market situation. Your report should be limited "
        f"to five sentences, yet comprehensive, highlighting key trends (e.g. pricing momentum, "
        f"supply and demand balance, market tightness) and considering their potential impact on home "
        f"prices in this region over the coming months."
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# P2. Prediction based on time series ("only time series" mode)
# ---------------------------------------------------------------------------

def predict_time_prompt(cfg, region_id, window, forecast_dates):
    label = region_label(region_id)
    system_prompt = (
        f"Your job is to act as a professional regional real-estate forecaster. You will be given "
        f"time-series data of housing market indicators from the past {cfg.lookback} months for "
        f"{label}. Based on this information, your task is to forecast the {target_label(cfg)} for "
        f"each of the next {cfg.horizon} months."
    )
    user_prompt = (
        f"Your task is to forecast the {target_label(cfg)} in {label} for each of the next "
        f"{cfg.horizon} months. Review the time-series data provided for the last {cfg.lookback} "
        f"months. Each time-series consists of monthly values separated by a '|' token for the "
        f"following indicators:\n\n"
        f"{indicator_block(cfg, window)}\n\n"
        f"Based on this information, {forecast_instruction(cfg, forecast_dates)}"
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# P3. Prediction based on text ("TimeCP" mode)
#
# The P1 report is deliberately non-numeric ("do not write numerical
# values"), which is fine for the paper's classification tasks (direction /
# threshold only) but starves a *regression* forecast of any sense of scale.
# So here the text summary is paired with the same raw indicator block P2
# uses, giving the model both the qualitative narrative and the numbers it
# needs to anchor a multi-step numeric forecast.
# ---------------------------------------------------------------------------

def predict_text_prompt(cfg, region_id, window, text, forecast_dates):
    label = region_label(region_id)
    system_prompt = (
        f"Your job is to act as a professional regional real-estate forecaster. You will be given "
        f"time-series data and a written summary of the housing market over the past {cfg.lookback} "
        f"months for {label}. Based on this information, your task is to forecast the "
        f"{target_label(cfg)} for each of the next {cfg.horizon} months."
    )
    user_prompt = (
        f"Your task is to forecast the {target_label(cfg)} in {label} for each of the next "
        f"{cfg.horizon} months. Review the {target_label(cfg)} data provided for the last {cfg.lookback} "
        f"months. The historical time series consists of monthly values separated by a '|' token: \n\n"
        f"{indicator_block(cfg, window, target_only=True)}\n\n"
        f"In addition, the housing market situation of the last {cfg.lookback} months is summarized "
        f"as follows:\n\n{text}\n\n"
        f"Based on both the time-series data and the summary above, {forecast_instruction(cfg, forecast_dates)}"
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# P4. Prediction of TimeCAP (time series + text + in-context retrieved examples)
#
# Same rationale as P3: both the query window and every retrieved example
# carry their raw lookback series next to the text, so the model has a
# numeric scale to calibrate against, not just a qualitative narrative plus
# a single outcome number.
# ---------------------------------------------------------------------------

def predict_in_context_prompt(cfg, region_id, window, text, forecast_dates, examples):
    label = region_label(region_id)
    k = len(examples)
    system_prompt = (
        f"Your job is to act as a professional regional real-estate forecaster. You will be given "
        f"time-series data and a written summary of the housing market over the past {cfg.lookback} "
        f"months for {label}. Based on this information, your task is to forecast the "
        f"{target_label(cfg)} for each of the next {cfg.horizon} months."
    )

    parts = [
        f"Your task is to forecast the {target_label(cfg)} in {label} for each of the next "
        f"{cfg.horizon} months.",
        f"First, review the following {k} examples of housing market time-series data and summaries "
        f"from other periods in this region's own history, and their actual {cfg.horizon}-month "
        f"outcomes, so you can refer to them when forecasting.\n",
    ]
    for idx, ex in enumerate(examples, 1):
        parts.append(f"Example #{idx} time-series (last {cfg.lookback} months):\n{ex['series_block']}")
        parts.append(f"Example #{idx} summary: {ex['text']}")
        parts.append(f"Example #{idx} outcome (next {cfg.horizon} months, {target_label(cfg)}): {ex['outcome']}\n")

    parts.append(
        f"The current time-series data for the last {cfg.lookback} months is as follows:\n"
        f"{indicator_block(cfg, window)}\n\n"
        f"The housing market situation of the last {cfg.lookback} months is summarized as follows:\n\n"
        f"Summary: {text}\nOutcome:\n\n"
        f"Refer to the provided examples and {forecast_instruction(cfg, forecast_dates)}"
    )
    user_prompt = '\n'.join(parts)
    return system_prompt, user_prompt
