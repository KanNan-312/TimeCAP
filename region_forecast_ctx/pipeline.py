import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from region_forecast.metrics import compute_metrics
from region_forecast.utils import ensure_dir, read_json, read_text, write_json, write_text

from region_forecast_ctx import data as D
from region_forecast_ctx import parsing as ctx_parsing
from region_forecast_ctx import prompts
from region_forecast_ctx.features import compute_target_features
from region_forecast_ctx.features.spatial_features import SpatialIndex, compute_neighbor_context


# ---------------------------------------------------------------------------
# Checkpoint file paths - same resumability pattern as the baseline pipeline:
# every stage checks for its own output file before doing (expensive) work.
# ---------------------------------------------------------------------------

def region_dir(cfg, region_id):
    return os.path.join(cfg.results_dir, cfg.mode, cfg.ablation_tag, str(region_id))


def summary_path(cfg, region_id, target_start):
    return os.path.join(region_dir(cfg, region_id), 'summaries', f'{target_start}.txt')


def features_path(cfg, region_id, target_start):
    return os.path.join(region_dir(cfg, region_id), 'features', f'{target_start}.json')


def prediction_path(cfg, region_id, target_start):
    return os.path.join(region_dir(cfg, region_id), 'predictions', f'{target_start}.json')


def meta_path(cfg, region_id):
    return os.path.join(region_dir(cfg, region_id), 'region_meta.json')


def metrics_path(cfg, region_id):
    return os.path.join(region_dir(cfg, region_id), 'metrics.json')


def experience_store_prefix(cfg):
    """Global (not per-region) - the store is fit once over the whole cross-region training pool."""
    return os.path.join(cfg.results_dir, cfg.mode, cfg.ablation_tag, 'experience_store')


# ---------------------------------------------------------------------------
# Stage 0: plan - cheap, deterministic, always (re)computed.
#
# Train-pool windows are always planned (not just when experience retrieval
# is on) since they're free to compute and give features/experience.py a
# ready-made pool to fit() against whenever it's implemented; a region is
# only *skipped* for lacking a train pool if experience retrieval is
# actually enabled for this run.
# ---------------------------------------------------------------------------

def plan_regions(cfg, df):
    indicator_cols = D.resolve_indicator_columns(df, cfg)

    plans = {}
    for region_id in D.list_region_ids(df, cfg):
        frame = D.get_region_frame(df, cfg, region_id)
        T = len(frame)
        min_len = cfg.lookback + cfg.horizon
        if T < min_len + 1:
            plan = {'status': 'skipped', 'reason': f'series too short ({T} rows, need >= {min_len + 1})'}
            plans[region_id] = plan
            write_json(meta_path(cfg, region_id), plan)
            continue

        starts = D.compute_target_starts(T, cfg, need_train_pool=True)
        if not starts['test_starts']:
            plan = {'status': 'skipped', 'reason': 'no valid test windows for the given split/lookback/horizon'}
            plans[region_id] = plan
            write_json(meta_path(cfg, region_id), plan)
            continue
        if cfg.experience_enabled and not starts['train_starts']:
            plan = {'status': 'skipped', 'reason': 'no valid train windows available for experience retrieval pool'}
            plans[region_id] = plan
            write_json(meta_path(cfg, region_id), plan)
            continue

        city = frame[cfg.city_column].iloc[0] if cfg.city_column in frame.columns else ''
        metro = frame[cfg.metro_column].iloc[0] if cfg.metro_column in frame.columns else ''
        plan = {'status': 'ok', 'T': T, 'city': city, 'metro': metro, **starts}
        plans[region_id] = plan
        write_json(meta_path(cfg, region_id), plan)

    return plans, indicator_cols


def _run_tasks(tasks, worker, workers, desc):
    if not tasks:
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
            task = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                print(f'[{desc}] FAILED task={task[:2]}: {e}')


def _ok_frames(cfg, df, plans):
    return {rid: D.get_region_frame(df, cfg, rid) for rid, p in plans.items() if p['status'] == 'ok'}


def build_spatial_index(cfg, df, plans):
    if cfg.context_level < 2 or not cfg.ctx_spatial:
        return None
    region_ids = [rid for rid, p in plans.items() if p['status'] == 'ok']
    locations = D.region_locations(df, cfg, region_ids)
    if len(locations) < 2:
        print('[spatial] fewer than 2 planned regions have valid lat/lon; spatial context disabled for this run')
        return None
    print(f'[spatial] distance index built over {len(locations)} region(s)')
    return SpatialIndex(locations)


# ---------------------------------------------------------------------------
# Stage 1: contextualize (P1) - skipped for mode == 'timeseries'.
#
# Deterministic, no LLM call: the feature calculator's output is serialized
# straight into text (data -> prompt -> prediction, see prompts.py). This
# also means the stage needs no API key and costs nothing to (re)run.
# ---------------------------------------------------------------------------

def run_contextualize_stage(cfg, df, plans, indicator_cols, spatial_index, experience_store):
    if cfg.mode == 'timeseries':
        print('[contextualize] skipped (timeseries mode forecasts directly from numeric series)')
        return

    if cfg.experience_enabled and not experience_store._fitted:
        # Lets a standalone `--stage contextualize` run (without `experience`
        # or `all` in the same invocation) pick up a store fitted earlier.
        prefix = experience_store_prefix(cfg)
        if os.path.exists(prefix + '.json'):
            experience_store.load(prefix)
            print(f'[contextualize] loaded cached experience store ({len(experience_store)} case(s), '
                  f'method={experience_store.method}) from {prefix}.json')
        else:
            print(f'[contextualize] WARNING: experience retrieval enabled but no fitted store found at '
                  f'{prefix}.json - run --stage experience (or --stage all) first. Precedent retrieval '
                  f'will return no matches until then.')

    frames = _ok_frames(cfg, df, plans)
    date_idx = {}
    if spatial_index is not None:
        date_idx = {rid: {d: i for i, d in enumerate(f[cfg.date_column].dt.strftime('%Y-%m-%d'))}
                    for rid, f in frames.items()}

    tasks = []
    for region_id, plan in plans.items():
        if plan['status'] != 'ok':
            continue
        for start in sorted(plan['test_starts']):
            if os.path.exists(summary_path(cfg, region_id, start)):
                continue
            tasks.append((region_id, start))

    print(f'[contextualize] {len(tasks)} pending window(s) (context_level={cfg.context_level}, deterministic - no LLM)')

    def worker(task):
        region_id, start = task
        frame = frames[region_id]
        history = D.extract_history(frame, cfg, indicator_cols, start)

        feats = compute_target_features(cfg, history)

        neighbor_ctx = None
        if cfg.spatial_enabled and spatial_index is not None:
            neighbor_ctx = compute_neighbor_context(cfg, region_id, start, frames, date_idx, spatial_index)

        experience_cases = None
        if cfg.experience_enabled:
            experience_cases = experience_store.query(region_id, start, history, cfg.experience_k)

        write_json(features_path(cfg, region_id, start), {
            'target': feats, 'neighbors': neighbor_ctx, 'experience': experience_cases,
        })

        text = prompts.build_context_text(cfg, feats, neighbor_ctx, experience_cases)
        write_text(summary_path(cfg, region_id, start), text)

    _run_tasks(tasks, worker, cfg.workers, 'contextualize')


# ---------------------------------------------------------------------------
# Stage 2: experience - fits ExperienceStore over the cross-region training
# pool (see features/experience.py for the retrieval methods). Must run
# before Stage 1's queries can return anything; persisted to disk so a
# standalone `--stage contextualize` run can reuse a prior fit without
# refitting (run_contextualize_stage loads it lazily if missing here).
# Skipped unless experience retrieval is enabled.
# ---------------------------------------------------------------------------

def run_experience_stage(cfg, df, plans, indicator_cols, experience_store):
    if not cfg.experience_enabled:
        print('[experience] skipped (context_level < 3 or ctx_experience=False)')
        return

    prefix = experience_store_prefix(cfg)
    if os.path.exists(prefix + '.json'):
        experience_store.load(prefix)
        print(f'[experience] loaded cached store ({len(experience_store)} case(s), '
              f'method={experience_store.method}) from {prefix}.json')
        return

    frames = _ok_frames(cfg, df, plans)
    train_cases = []
    for region_id, plan in plans.items():
        if plan['status'] != 'ok':
            continue
        frame = frames[region_id]
        dates_str = frame[cfg.date_column].dt.strftime('%Y-%m-%d')
        for start in plan['train_starts']:
            window = D.extract_window(frame, cfg, indicator_cols, start - cfg.lookback, cfg.lookback)
            history = D.extract_history(frame, cfg, indicator_cols, start)
            outcome = D.extract_window(frame, cfg, indicator_cols, start, cfg.horizon)
            # Real calendar date of the last known month for this case -
            # kept only for internal traceback (checkpoint files / store),
            # never surfaced in LLM-facing text (see features/experience.py).
            anchor_date = dates_str.iloc[start - 1] if start > 0 else None
            train_cases.append({
                'region_id': region_id, 'start': start, 'anchor_date': anchor_date,
                'window': window, 'history': history, 'outcome': outcome['target'],
            })

    experience_store.fit(train_cases)
    experience_store.save(prefix)
    print(f'[experience] fitted store (method={cfg.experience_method}) over {len(train_cases)} training-pool '
          f'case(s), {len(experience_store)} retained after outcome filtering -> {prefix}.*')


# ---------------------------------------------------------------------------
# Stage 3: predict (P2 for timeseries mode / P3 for timecp_ctx) - test
# windows only. Prompt shape here is identical to the baseline TimeCP mode
# except for two enrichments (both region_forecast_ctx-only - the baseline
# templates/parser are untouched):
#   - the LLM is asked to return a small JSON object (prediction + rationale
#     + which categories of information it says it used) instead of a bare
#     '|'-separated number string, so a forecast's reasoning and claimed
#     information sources are inspectable, not just its numbers;
#   - if ctx_correlation found indicator(s) strongly correlated with the
#     target (see features/target_features.compute_correlation), their own
#     lookback series is shown alongside the target's, not just the
#     correlation-coefficient sentence in the P1 text.
# ---------------------------------------------------------------------------

def _correlated_indicator_names(cfg, region_id, start):
    """
    Indicator names the contextualize stage found strongly correlated with
    the target for this specific window (see target_features.compute_correlation),
    read back from the features/*.json checkpoint it wrote. None if
    unavailable (mode == 'timeseries', ctx_correlation/ctx_show_correlated_series
    off, or the contextualize stage hasn't run yet for this window).
    """
    if cfg.mode != 'timecp_ctx' or not cfg.ctx_correlation or not cfg.ctx_show_correlated_series:
        return None
    fpath = features_path(cfg, region_id, start)
    if not os.path.exists(fpath):
        return None
    feats = read_json(fpath)
    corr = ((feats.get('target') or {}).get('correlation') or {})
    names = [name for name, _r in corr.get('correlations', [])]
    return names or None


def run_predict_stage(cfg, df, plans, indicator_cols, llm):
    frames = _ok_frames(cfg, df, plans)

    tasks = []
    for region_id, plan in plans.items():
        if plan['status'] != 'ok':
            continue
        for start in plan['test_starts']:
            if os.path.exists(prediction_path(cfg, region_id, start)):
                continue
            tasks.append((region_id, start))

    print(f'[predict] {len(tasks)} pending window(s)')

    def worker(task):
        region_id, start = task
        frame = frames[region_id]
        window = D.extract_window(frame, cfg, indicator_cols, start - cfg.lookback, cfg.lookback)
        forecast_window = D.extract_window(frame, cfg, indicator_cols, start, cfg.horizon)
        forecast_dates = forecast_window['dates']
        true_values = forecast_window['target']

        if cfg.mode == 'timeseries':
            system_prompt, user_prompt = prompts.predict_time_prompt_structured(cfg, region_id, window, forecast_dates)
        else:  # timecp_ctx
            # build_context_text() produces a structured, section-headed
            # block (not free-form prose like the baseline's LLM report),
            # so its blank lines are preserved rather than collapsed.
            text = read_text(summary_path(cfg, region_id, start)).strip()
            correlated_names = _correlated_indicator_names(cfg, region_id, start)
            system_prompt, user_prompt = prompts.predict_text_prompt_structured(
                cfg, region_id, window, text, forecast_dates, correlated_names)

        raw = llm.chat(system_prompt, user_prompt, expect_numeric=True, n_values=cfg.horizon)
        pred, rationale, information_source, parse_error = ctx_parsing.parse_structured_forecast(
            raw, cfg.horizon, expect_json=cfg.predict_thinking)

        record = {
            'region_id': region_id,
            'target_start': start,
            'lookback_dates': window['dates'],
            'forecast_dates': forecast_dates,
            'raw_response': raw,
            'pred': pred,
            'true': true_values,
            'parse_error': parse_error,
        }
        if cfg.predict_thinking:
            record['rationale'] = rationale
            record['information_source'] = information_source
        write_json(prediction_path(cfg, region_id, start), record)

    _run_tasks(tasks, worker, cfg.workers, 'predict')


# ---------------------------------------------------------------------------
# Stage 4: evaluate - cheap and deterministic, always recomputed.
# ---------------------------------------------------------------------------

def run_evaluate_stage(cfg, plans):
    rows = []
    all_preds, all_trues = [], []

    for region_id, plan in plans.items():
        row = {'region_id': region_id, 'status': plan['status']}
        if plan['status'] != 'ok':
            row['reason'] = plan.get('reason', '')
            rows.append(row)
            continue

        preds, trues, n_missing, n_parse_fail = [], [], 0, 0
        for start in plan['test_starts']:
            path = prediction_path(cfg, region_id, start)
            if not os.path.exists(path):
                n_missing += 1
                continue
            rec = read_json(path)
            if rec['pred'] is None:
                n_parse_fail += 1
                continue
            preds.extend(rec['pred'])
            trues.extend(rec['true'])

        row['n_test_windows'] = len(plan['test_starts'])
        row['n_missing_predictions'] = n_missing
        row['n_parse_failures'] = n_parse_fail

        if preds:
            m = compute_metrics(preds, trues)
            row.update(m)
            write_json(metrics_path(cfg, region_id), m)
            all_preds.extend(preds)
            all_trues.extend(trues)
        else:
            row['status'] = 'no_valid_predictions'
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(cfg.results_dir, cfg.mode, cfg.ablation_tag, 'region_metrics.csv')
    ensure_dir(os.path.dirname(summary_csv))
    summary_df.to_csv(summary_csv, index=False)

    overall = compute_metrics(all_preds, all_trues) if all_preds else {}
    overall['n_regions_planned'] = len(plans)
    overall['n_regions_with_valid_predictions'] = int((summary_df['status'] == 'ok').sum()) if len(summary_df) else 0
    overall['context_level'] = cfg.context_level
    overall['ablation_tag'] = cfg.ablation_tag
    write_json(os.path.join(cfg.results_dir, cfg.mode, cfg.ablation_tag, 'overall_metrics.json'), overall)

    print(f'[evaluate] region metrics -> {summary_csv}')
    print(f'[evaluate] overall test-set metrics ({cfg.mode}, {cfg.ablation_tag}): {overall}')
    return summary_df, overall
