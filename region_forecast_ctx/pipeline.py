import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from region_forecast import parsing
from region_forecast.metrics import compute_metrics
from region_forecast.utils import ensure_dir, read_json, read_text, write_json, write_text

from region_forecast_ctx import data as D
from region_forecast_ctx import prompts
from region_forecast_ctx.features import compute_target_features
from region_forecast_ctx.features.experience import ExperienceStore
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
# ---------------------------------------------------------------------------

def run_contextualize_stage(cfg, df, plans, indicator_cols, llm, spatial_index, experience_store):
    if cfg.mode == 'timeseries':
        print('[contextualize] skipped (timeseries mode forecasts directly from numeric series)')
        return

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

    print(f'[contextualize] {len(tasks)} pending window(s) (context_level={cfg.context_level})')

    def worker(task):
        region_id, start = task
        frame = frames[region_id]
        window = D.extract_window(frame, cfg, indicator_cols, start - cfg.lookback, cfg.lookback)
        history = D.extract_history(frame, cfg, indicator_cols, start)

        feats = compute_target_features(cfg, window, history)

        neighbor_ctx = None
        if cfg.spatial_enabled and spatial_index is not None:
            neighbor_ctx = compute_neighbor_context(cfg, region_id, start, frames, date_idx, spatial_index)

        experience_cases = None
        if cfg.experience_enabled:
            experience_cases = experience_store.query(region_id, window, cfg.experience_k)

        write_json(features_path(cfg, region_id, start), {
            'target': feats, 'neighbors': neighbor_ctx, 'experience': experience_cases,
        })

        system_prompt, user_prompt = prompts.contextualize_prompt(
            cfg, region_id, window, feats, neighbor_ctx, experience_cases)
        text = llm.chat(system_prompt, user_prompt)
        write_text(summary_path(cfg, region_id, start), text)

    _run_tasks(tasks, worker, cfg.workers, 'contextualize')


# ---------------------------------------------------------------------------
# Stage 2: experience - fits the (currently placeholder) ExperienceStore
# over each region's training pool. A no-op today; wired up so
# context_level=3 exercises the full pipeline shape once query() does
# something. Skipped unless experience retrieval is enabled.
# ---------------------------------------------------------------------------

def run_experience_stage(cfg, df, plans, indicator_cols, experience_store):
    if not cfg.experience_enabled:
        print('[experience] skipped (context_level < 3 or ctx_experience=False)')
        return

    frames = _ok_frames(cfg, df, plans)
    train_cases = []
    for region_id, plan in plans.items():
        if plan['status'] != 'ok':
            continue
        frame = frames[region_id]
        for start in plan['train_starts']:
            window = D.extract_window(frame, cfg, indicator_cols, start - cfg.lookback, cfg.lookback)
            history = D.extract_history(frame, cfg, indicator_cols, start)
            outcome = D.extract_window(frame, cfg, indicator_cols, start, cfg.horizon)
            train_cases.append({
                'region_id': region_id, 'start': start,
                'window': window, 'history': history, 'outcome': outcome['target'],
            })

    experience_store.fit(train_cases)
    print(f'[experience] fitted placeholder store over {len(train_cases)} training-pool case(s) '
          f'(retrieval not yet implemented - see features/experience.py)')


# ---------------------------------------------------------------------------
# Stage 3: predict (P2 for timeseries mode / P3 for timecp_ctx) - test
# windows only. Prompt shape here is identical to the baseline TimeCP mode;
# the ablation only changes what the P1 summary text (fed in as `text`)
# contains.
# ---------------------------------------------------------------------------

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
            system_prompt, user_prompt = prompts.predict_time_prompt(cfg, region_id, window, forecast_dates)
        else:  # timecp_ctx
            text = read_text(summary_path(cfg, region_id, start)).replace('\n\n', ' ').strip()
            system_prompt, user_prompt = prompts.predict_text_prompt(cfg, region_id, window, text, forecast_dates)

        raw = llm.chat(system_prompt, user_prompt, expect_numeric=True, n_values=cfg.horizon)
        pred, parse_error = parsing.parse_forecast(raw, cfg.horizon)

        write_json(prediction_path(cfg, region_id, start), {
            'region_id': region_id,
            'target_start': start,
            'lookback_dates': window['dates'],
            'forecast_dates': forecast_dates,
            'raw_response': raw,
            'pred': pred,
            'true': true_values,
            'parse_error': parse_error,
        })

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
