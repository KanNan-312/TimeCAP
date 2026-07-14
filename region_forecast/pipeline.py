import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from region_forecast import data as D
from region_forecast import prompts
from region_forecast import parsing
from region_forecast.embedding_store import embed_texts, top_k_indices
from region_forecast.metrics import compute_metrics
from region_forecast.utils import ensure_dir, read_json, read_text, write_json, write_text


# ---------------------------------------------------------------------------
# Checkpoint file paths - every stage below is resumable purely because it
# checks for these files before doing any (expensive) work.
# ---------------------------------------------------------------------------

def region_dir(cfg, region_id):
    return os.path.join(cfg.results_dir, cfg.mode, str(region_id))


def summary_path(cfg, region_id, target_start):
    return os.path.join(region_dir(cfg, region_id), 'summaries', f'{target_start}.txt')


def prediction_path(cfg, region_id, target_start):
    return os.path.join(region_dir(cfg, region_id), 'predictions', f'{target_start}.json')


def embeddings_path(cfg, region_id):
    return os.path.join(region_dir(cfg, region_id), 'train_embeddings.npz')


def meta_path(cfg, region_id):
    return os.path.join(region_dir(cfg, region_id), 'region_meta.json')


def metrics_path(cfg, region_id):
    return os.path.join(region_dir(cfg, region_id), 'metrics.json')


# ---------------------------------------------------------------------------
# Stage 0: plan - cheap, deterministic, always (re)computed; not gated by
# resume logic since it involves no LLM calls.
# ---------------------------------------------------------------------------

def plan_regions(cfg, df):
    indicator_cols = D.resolve_indicator_columns(df, cfg)
    need_train_pool = cfg.mode == 'timecap'

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

        starts = D.compute_target_starts(T, cfg, need_train_pool)
        if not starts['test_starts']:
            plan = {'status': 'skipped', 'reason': 'no valid test windows for the given split/lookback/horizon'}
            plans[region_id] = plan
            write_json(meta_path(cfg, region_id), plan)
            continue
        if need_train_pool and not starts['train_starts']:
            plan = {'status': 'skipped', 'reason': 'no valid train windows available for in-context retrieval pool'}
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


# ---------------------------------------------------------------------------
# Stage 1: contextualize (P1) - skipped entirely for mode == 'timeseries'.
# ---------------------------------------------------------------------------

def run_contextualize_stage(cfg, df, plans, indicator_cols, llm):
    if cfg.mode == 'timeseries':
        print('[contextualize] skipped (timeseries mode forecasts directly from numeric series)')
        return

    tasks = []
    for region_id, plan in plans.items():
        if plan['status'] != 'ok':
            continue
        frame = D.get_region_frame(df, cfg, region_id)
        starts = set(plan['test_starts'])
        if cfg.mode == 'timecap':
            starts |= set(plan['train_starts'])
        for start in sorted(starts):
            if os.path.exists(summary_path(cfg, region_id, start)):
                continue
            tasks.append((region_id, start, frame))

    print(f'[contextualize] {len(tasks)} pending window(s)')

    def worker(task):
        region_id, start, frame = task
        window = D.extract_window(frame, cfg, indicator_cols, start - cfg.lookback, cfg.lookback)
        system_prompt, user_prompt = prompts.contextualize_prompt(cfg, region_id, window)
        text = llm.chat(system_prompt, user_prompt)
        write_text(summary_path(cfg, region_id, start), text)

    _run_tasks(tasks, worker, cfg.workers, 'contextualize')


# ---------------------------------------------------------------------------
# Stage 2: embed train-pool summaries (TimeCAP mode only).
# ---------------------------------------------------------------------------

def run_embed_stage(cfg, plans):
    if cfg.mode != 'timecap':
        print('[embed] skipped (only used by timecap mode)')
        return

    for region_id, plan in plans.items():
        if plan['status'] != 'ok':
            continue
        out_path = embeddings_path(cfg, region_id)
        if os.path.exists(out_path):
            continue

        texts, valid_starts = [], []
        for start in plan['train_starts']:
            sp = summary_path(cfg, region_id, start)
            if not os.path.exists(sp):
                continue  # contextualize stage hasn't produced this one yet
            texts.append(read_text(sp))
            valid_starts.append(start)

        if not texts:
            continue

        vecs = embed_texts(texts, cfg)
        ensure_dir(os.path.dirname(out_path))
        np.savez(out_path, starts=np.array(valid_starts), vecs=vecs)

    print('[embed] done')


# ---------------------------------------------------------------------------
# Stage 3: predict (P2 / P3 / P4 depending on mode) - only on test windows.
# ---------------------------------------------------------------------------

def run_predict_stage(cfg, df, plans, indicator_cols, llm):
    tasks = []
    for region_id, plan in plans.items():
        if plan['status'] != 'ok':
            continue
        frame = D.get_region_frame(df, cfg, region_id)
        for start in plan['test_starts']:
            if os.path.exists(prediction_path(cfg, region_id, start)):
                continue
            tasks.append((region_id, start, frame))

    print(f'[predict] {len(tasks)} pending window(s)')

    train_pool_cache = {}

    def load_train_pool(region_id):
        if region_id not in train_pool_cache:
            npz = np.load(embeddings_path(cfg, region_id), allow_pickle=True)
            starts = npz['starts'].tolist()
            vecs = npz['vecs']
            texts = {s: read_text(summary_path(cfg, region_id, s)) for s in starts}
            train_pool_cache[region_id] = (starts, vecs, texts)
        return train_pool_cache[region_id]

    def worker(task):
        region_id, start, frame = task
        window = D.extract_window(frame, cfg, indicator_cols, start - cfg.lookback, cfg.lookback)
        forecast_window = D.extract_window(frame, cfg, indicator_cols, start, cfg.horizon)
        forecast_dates = forecast_window['dates']
        true_values = forecast_window['target']

        if cfg.mode == 'timeseries':
            system_prompt, user_prompt = prompts.predict_time_prompt(
                cfg, region_id, window, forecast_dates)
        else:
            text = read_text(summary_path(cfg, region_id, start)).replace('\n\n', ' ').strip()
            if cfg.mode == 'timecp':
                system_prompt, user_prompt = prompts.predict_text_prompt(
                    cfg, region_id, text, forecast_dates)
            else:  # timecap
                starts, vecs, texts = load_train_pool(region_id)
                query_vec = embed_texts([text], cfg)[0]
                order, _sims = top_k_indices(query_vec, vecs, min(cfg.k_examples, len(starts)))
                examples = []
                for oi in order:
                    ex_start = starts[oi]
                    ex_forecast = D.extract_window(frame, cfg, indicator_cols, ex_start, cfg.horizon)
                    examples.append({
                        'text': texts[ex_start].replace('\n\n', ' ').strip(),
                        'outcome': prompts.format_series(ex_forecast['target']),
                    })
                system_prompt, user_prompt = prompts.predict_in_context_prompt(
                    cfg, region_id, text, forecast_dates, examples)

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
    summary_csv = os.path.join(cfg.results_dir, cfg.mode, 'region_metrics.csv')
    ensure_dir(os.path.dirname(summary_csv))
    summary_df.to_csv(summary_csv, index=False)

    overall = compute_metrics(all_preds, all_trues) if all_preds else {}
    overall['n_regions_planned'] = len(plans)
    overall['n_regions_with_valid_predictions'] = int((summary_df['status'] == 'ok').sum()) if len(summary_df) else 0
    write_json(os.path.join(cfg.results_dir, cfg.mode, 'overall_metrics.json'), overall)

    print(f"[evaluate] region metrics -> {summary_csv}")
    print(f"[evaluate] overall test-set metrics ({cfg.mode}): {overall}")
    return summary_df, overall
